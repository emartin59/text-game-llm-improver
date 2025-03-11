import random
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Define quantization configuration for 8-bit precision
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    bnb_8bit_compute_dtype=torch.float16  # Use FP16 for computations
)

# Load tokenizer and initial control model with INT8 precision
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", padding_side='left')
control_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=quant_config
)

# Verify padding side
print(f"Tokenizer padding side: {tokenizer.padding_side}")

# Global variables for bot IDs and direct clones tracking
next_id = 4  # Start at 4 since we have 4 initial bots (IDs 0-3)
direct_clones = {}

def sample_x_percent():
    """
    Sample x% where 50% of results are between 0.0% and 1.0%, and 50% are between 1.0% and 20.0%.
    Using uniform sampling within each range for simplicity.
    """
    if random.random() < 0.5:
        return random.uniform(0.0, 1.0)
    else:
        return random.uniform(1.0, 20.0)

def create_modified_model(control_model, x):
    """
    Create a modified version of the control model by altering x% of its floating-point parameters.
    
    Args:
        control_model: The original model to modify.
        x: Percentage of parameters to modify (0-100).
    Returns:
        A new model with modified floating-point parameters.
    """
    # Clone the state dictionary to work with a copy of the parameters
    variable_state_dict = {k: v.clone().to('cpu') for k, v in control_model.state_dict().items()}
    
    # Modify x% of the floating-point parameters
    for name, p in variable_state_dict.items():
        # Check if the parameter is a floating-point type
        if p.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            num_to_modify = round(p.numel() * x / 100)  # Number of elements to modify
            if num_to_modify > 0:
                # Randomly select indices to modify
                indices = torch.randperm(p.numel())[:num_to_modify]
                # Assign random values only to floating-point parameters
                p.view(-1)[indices] = torch.randn(num_to_modify, dtype=p.dtype, device=p.device) * 0.02
        else:
            # Optional: Log skipped parameters for debugging
            print(f"Skipping modification for parameter '{name}' with dtype {p.dtype}")
    
    # Load a new instance of the quantized model
    variable_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quantization_config=quant_config  # Ensure this matches your original setup
    )
    # Apply the modified state dictionary
    variable_model.load_state_dict(variable_state_dict)
    
    return variable_model

def tinyllama_query(prompts, model):
    """Batch process prompts through the specified model."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=1.0,
        top_k=10,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    input_lengths = inputs['attention_mask'].sum(dim=1)
    responses = [tokenizer.decode(outputs[i, input_lengths[i]:], skip_special_tokens=True) for i in range(len(prompts))]
    return responses

def get_action_from_response(response, bot, terrain, bots):
    """Parse model response into an action with hierarchical matching."""
    response = response.lower().strip()
    words = response.split()  # Split into words

    # Define command keywords and their two-character prefixes
    commands = [
        ("mate", "ma", "m", lambda: bot['energy'] >= 75 and any(abs(other['y'] - bot['y']) + abs(other['x'] - bot['x']) == 1 for other in bots if other != bot)),
        ("up", "up", "u", lambda: True),
        ("down", "do", "d", lambda: True),
        ("left", "le", "l", lambda: True),
        ("right", "ri", "r", lambda: True),
        ("eat", "ea", "e", lambda: bot['seeds'] >= 10),
        ("plant", "pl", "p", lambda: bot['seeds'] > 0 and terrain[bot['y']][bot['x']] == '0'),
        ("talk:", "ta", "t", lambda: True)  # "talk:" is a prefix, handled specially
    ]

    # Step 1: Exact standalone word matches
    for word in reversed(words):
        for cmd, _, _, condition in commands:
            if cmd == "talk:" and word.startswith("talk:") and condition():
                message = response[len("talk:"):].strip()
                return ("Talk", message)
            elif word == cmd and condition():
                return cmd.capitalize() if cmd != "talk:" else cmd

    # Step 2: Keywords anywhere within words
    for word in reversed(words):
        for cmd, _, _, condition in commands:
            if cmd == "talk:" and word.startswith("talk:") and condition():
                message = response[len("talk:"):].strip()
                return ("Talk", message)
            elif cmd in word and condition():
                return cmd.capitalize() if cmd != "talk:" else cmd

    # Step 3: First two characters at the start of a word
    for word in reversed(words):
        if len(word) >= 2:
            two_chars = word[:2]
            for cmd, two_prefix, _, condition in commands:
                if two_chars == two_prefix and condition():
                    if cmd == "talk:":
                        message = response.strip()
                        return ("Talk", message)
                    return cmd.capitalize()

    # Step 4: First two characters anywhere in a word
    for word in reversed(words):
        for cmd, two_prefix, _, condition in commands:
            if two_prefix in word and condition():
                if cmd == "talk:":
                    message = response.strip()
                    return ("Talk", message)
                return cmd.capitalize()

    # Step 5: Single letter at the start of a word
    for word in reversed(words):
        if len(word) > 0:
            first_letter = word[0]
            for cmd, _, letter, condition in commands:
                if first_letter == letter and condition():
                    if cmd == "talk:":
                        message = response.strip()
                        return ("Talk", message)
                    return cmd.capitalize()

    # Step 6: Single letter anywhere in a word
    for word in reversed(words):
        for cmd, _, letter, condition in commands:
            if letter in word and condition():
                if cmd == "talk:":
                    message = response.strip()
                    return ("Talk", message)
                return cmd.capitalize()

    # Final fallback
    return choose_fallback_action(bot, terrain, bots)

def choose_fallback_action(bot, terrain, bots):
    """Fallback action if model response is invalid."""
    y, x, energy, seeds = bot['y'], bot['x'], bot['energy'], bot['seeds']
    if energy >= 75 and any(abs(other['y'] - y) + abs(other['x'] - x) == 1 for other in bots if other != bot):
        return "Mate"
    if energy < 50 and seeds >= 10:
        return "Eat"
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_y, new_x = y + dy, x + dx
        if 0 <= new_y < 9 and 0 <= new_x < 9 and terrain[new_y][new_x] == 'w':
            return {(-1, 0): "Up", (1, 0): "Down", (0, -1): "Left", (0, 1): "Right"}[(dy, dx)]
    if seeds > 0 and terrain[y][x] == '0':
        return "Plant"
    return random.choice(["Up", "Down", "Left", "Right"])

def create_terrain():
    """Initialize the 9x9 terrain with walls, wheat, rocks, and 4 bot positions."""
    terrain = [['u' if y == 0 or y == 8 or x == 0 or x == 8 else '0' for x in range(9)] for y in range(9)]
    inner_positions = [(y, x) for y in range(1, 8) for x in range(1, 8)]
    random.shuffle(inner_positions)

    for i in range(20):  # 20 wheat
        y, x = inner_positions[i]
        terrain[y][x] = 'w'

    for i in range(20, 25):  # 5 rocks
        y, x = inner_positions[i]
        terrain[y][x] = 'r'

    available = [(y, x) for y in range(1, 8) for x in range(1, 8) if terrain[y][x] == '0']
    random.shuffle(available)
    bot_positions = available[:4] if len(available) >= 4 else [(1, 1), (1, 2), (7, 6), (7, 7)]  # Fallback to 4 positions

    return terrain, bot_positions

def is_occupied(y, x, bots):
    """Check if a position is occupied by a bot."""
    return any(bot['y'] == y and bot['x'] == x for bot in bots)

def get_bot_view(bot, bots, terrain):
    """Generate the bot's 9x9 view of the terrain."""
    view = [row[:] for row in terrain]
    for other_bot in bots:
        if other_bot == bot:
            view[other_bot['y']][other_bot['x']] = 'x'
        elif other_bot['is_talking']:
            view[other_bot['y']][other_bot['x']] = 't'
        else:
            view[other_bot['y']][other_bot['x']] = 'z'
    return '\n'.join(' '.join(row) for row in view)

def get_global_view(bots, terrain):
    """Generate a global view of the terrain with all bots marked."""
    view = [row[:] for row in terrain]
    for bot in bots:
        if bot['is_talking']:
            view[bot['y']][bot['x']] = 't'
        else:
            view[bot['y']][bot['x']] = 'z'
    return '\n'.join(' '.join(row) for row in view)

def get_prompt(bot, bots, terrain):
    """Generate the prompt for a bot."""
    map_view = get_bot_view(bot, bots, terrain)
    prompt = f"""
You have {bot['energy']} energy and {bot['seeds']} seeds. You are 'x' on the map:

{map_view}

'r' is rock (push if space behind is empty), 'z' is another bot, 't' is a talking bot, 'w' is wheat (move onto it for 10 seeds), '0' is empty, 'u' is a wall.

Goal: Be the first to have 3 direct clones via mating. Mating requires a bot ('z' or 't') next to you and 75 energy. Don’t run out of energy.

Commands:
- 'Mate': Clone if next to a bot (75 energy)
- 'Up', 'Down', 'Left', 'Right': Move
- 'Eat': 10 seeds -> 10 energy (1 energy, needs 10 seeds)
- 'Plant': Plant a seed on '0' (1 energy, 1 seed, grows to 'w' in 10 turns)
- 'Talk: [message]': Send a message (1 energy, no other action, shown as 't' until next turn)

One command only: Mate, Up, Down, Left, Right, Eat, Plant, or Talk: [message].
"""
    return prompt

def process_bot_action(bot, action, terrain, bots, planted, action_count):
    """Execute a bot's action."""
    global next_id, direct_clones
    if isinstance(action, tuple) and action[0] == "Talk":
        bot['is_talking'] = True
        bot['message'] = action[1]
        bot['energy'] -= 1
    elif action in ["Up", "Down", "Left", "Right"]:
        dy, dx = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}[action]
        new_y, new_x = bot['y'] + dy, bot['x'] + dx
        if 0 <= new_y < 9 and 0 <= new_x < 9:
            target = terrain[new_y][new_x]
            if target in ['0', 'v', 'w'] and not is_occupied(new_y, new_x, bots):
                if (bot['y'], bot['x']) in planted:
                    age = action_count - planted[(bot['y'], bot['x'])]
                    terrain[bot['y']][bot['x']] = 'w' if age >= 10 else 'v'
                bot['y'], bot['x'] = new_y, new_x
                if target == 'w':
                    bot['seeds'] += 10
                    terrain[new_y][new_x] = '0'
                    if (new_y, new_x) in planted:
                        del planted[(new_y, new_x)]
            elif target == 'r':
                rock_new_y = new_y + dy
                rock_new_x = new_x + dx
                if 0 <= rock_new_y < 9 and 0 <= rock_new_x < 9 and terrain[rock_new_y][rock_new_x] == '0':
                    terrain[rock_new_y][rock_new_x] = 'r'
                    terrain[new_y][new_x] = '0'
                    bot['y'], bot['x'] = new_y, new_x
        bot['energy'] -= 1
    elif action == "Eat" and bot['seeds'] >= 10:
        bot['seeds'] -= 10
        bot['energy'] = min(500, bot['energy'] + 10 - 1)
    elif action == "Plant" and bot['seeds'] > 0:
        if terrain[bot['y']][bot['x']] == '0' and (bot['y'], bot['x']) not in planted:
            planted[(bot['y'], bot['x'])] = action_count
            bot['seeds'] -= 1
        bot['energy'] -= 1
    elif action == "Mate" and bot['energy'] >= 75:
        adjacent = [other for other in bots if abs(other['y'] - bot['y']) + abs(other['x'] - bot['x']) == 1]
        if adjacent:
            positions = [(bot['y'] - 1, bot['x']), (bot['y'] + 1, bot['x']),
                         (bot['y'], bot['x'] - 1), (bot['y'], bot['x'] + 1)]
            valid = [pos for pos in positions if 0 <= pos[0] < 9 and 0 <= pos[1] < 9
                     and terrain[pos[0]][pos[1]] == '0' and not is_occupied(pos[0], pos[1], bots)]
            if valid:
                new_y, new_x = random.choice(valid)
                new_id = next_id
                next_id += 1
                new_bot = {
                    'id': new_id,
                    'y': new_y,
                    'x': new_x,
                    'energy': 100,
                    'seeds': bot['seeds'],
                    'last_action': None,
                    'is_talking': False,
                    'message': None,
                    'llm_type': bot['llm_type']
                }
                bots.append(new_bot)
                direct_clones[bot['id']] += 1
                direct_clones[new_id] = 0
                bot['energy'] -= 75

def run_game(control_model, variable_model):
    """Run a single game and return the winner ('control', 'variable', or 'tie')."""
    global next_id, direct_clones
    terrain, bot_positions = create_terrain()
    bots = [
        # Two control bots
        {
            'id': 0,
            'y': bot_positions[0][0],
            'x': bot_positions[0][1],
            'energy': 100,
            'seeds': 0,
            'last_action': None,
            'is_talking': False,
            'message': None,
            'llm_type': 'control'
        },
        {
            'id': 1,
            'y': bot_positions[1][0],
            'x': bot_positions[1][1],
            'energy': 100,
            'seeds': 0,
            'last_action': None,
            'is_talking': False,
            'message': None,
            'llm_type': 'control'
        },
        # Two variable bots
        {
            'id': 2,
            'y': bot_positions[2][0],
            'x': bot_positions[2][1],
            'energy': 100,
            'seeds': 0,
            'last_action': None,
            'is_talking': False,
            'message': None,
            'llm_type': 'variable'
        },
        {
            'id': 3,
            'y': bot_positions[3][0],
            'x': bot_positions[3][1],
            'energy': 100,
            'seeds': 0,
            'last_action': None,
            'is_talking': False,
            'message': None,
            'llm_type': 'variable'
        }
    ]
    next_id = 4  # Start after initial 4 bots
    direct_clones = {0: 0, 1: 0, 2: 0, 3: 0}
    action_count = 0
    planted = {}
    control_created = 2  # Initial control bots
    variable_created = 2  # Initial variable bots

    while True:
        # Collect output for this turn
        turn_output = [f"Turn {action_count + 1}:", "Global view:", get_global_view(bots, terrain), ""]

        active_bots = [bot for bot in bots if bot['energy'] > 0]
        control_bots = [bot for bot in active_bots if bot['llm_type'] == 'control']
        variable_bots = [bot for bot in active_bots if bot['llm_type'] == 'variable']

        # Process control bots
        control_prompts = [get_prompt(bot, bots, terrain) for bot in control_bots]
        control_responses = tinyllama_query(control_prompts, control_model) if control_prompts else []
        for bot, response in zip(control_bots, control_responses):
            action = get_action_from_response(response, bot, terrain, bots)
            turn_output.append(f"Bot {bot['id']} ({bot['llm_type']}) at ({bot['y']},{bot['x']}) with energy {bot['energy']}, seeds {bot['seeds']}: Action: {action}")
            turn_output.append(f"Raw response: '{response}'")
            bot['last_action'] = action
            if action == "Mate" and bot['energy'] >= 75:  # Check before processing to count only successful mates
                control_created += 1
            process_bot_action(bot, action, terrain, bots, planted, action_count)

        # Process variable bots
        variable_prompts = [get_prompt(bot, bots, terrain) for bot in variable_bots]
        variable_responses = tinyllama_query(variable_prompts, variable_model) if variable_prompts else []
        for bot, response in zip(variable_bots, variable_responses):
            action = get_action_from_response(response, bot, terrain, bots)
            turn_output.append(f"Bot {bot['id']} ({bot['llm_type']}) at ({bot['y']},{bot['x']}) with energy {bot['energy']}, seeds {bot['seeds']}: Action: {action}")
            turn_output.append(f"Raw response: '{response}'")
            bot['last_action'] = action
            if action == "Mate" and bot['energy'] >= 75:  # Check before processing to count only successful mates
                variable_created += 1
            process_bot_action(bot, action, terrain, bots, planted, action_count)

        # Update planted seeds
        for (y, x), count in list(planted.items()):
            if terrain[y][x] == 'v' and not is_occupied(y, x, bots):
                if action_count - count >= 10:
                    terrain[y][x] = 'w'

        # Remove dead bots
        bots = [bot for bot in bots if bot['energy'] > 0]

        # Check if game is ending
        game_ending = (direct_clones.get(0, 0) >= 3 or direct_clones.get(1, 0) >= 3 or 
                       direct_clones.get(2, 0) >= 3 or direct_clones.get(3, 0) >= 3 or 
                       len(bots) == 0 or next_id >= 24)  # 20 clones + 4 initial = 24

        # Print output if conditions met
        if action_count == 0 or action_count % 100 == 0 or game_ending:
            print("\n".join(turn_output))

        # Handle game ending and determine winner
        if game_ending:
            if control_created > variable_created:
                winner = 'control'
            elif variable_created > control_created:
                winner = 'variable'
            else:
                winner = 'tie'
            if direct_clones.get(0, 0) >= 3 or direct_clones.get(1, 0) >= 3:
                print("Game ended because a control bot has 3 or more direct clones.")
            elif direct_clones.get(2, 0) >= 3 or direct_clones.get(3, 0) >= 3:
                print("Game ended because a variable bot has 3 or more direct clones.")
            elif len(bots) == 0:
                print("Game ended because all bots died.")
            elif next_id >= 24:
                print("Game ended because 20 total bots have been cloned.")
            print(f"Winner: {winner} (control created: {control_created}, variable created: {variable_created})")
            return winner

        action_count += 1

def main():
    """Run the game loop continuously, evolving the control LLM based on game outcomes."""
    global control_model
    winner_count = 1

    # No need to move control_model to CUDA; it's already on the GPU due to quantization
    # Removed: if torch.cuda.is_available():
    #              control_model = control_model.to('cuda')

    while True:
        # Create a new variable LLM
        x = sample_x_percent()
        variable_model = create_modified_model(control_model, x)
        print(f"***Variable bot created as a {x:.2f}% clone of the control LLM***")

        # Run the first test game
        winner = run_game(control_model, variable_model)

        if winner == 'variable':
            wins = 1  # Count the first win
            for m in range(4):  # m from 0 to 3, representing additional games
                winner = run_game(control_model, variable_model)
                if winner == 'variable':
                    wins += 1
                # Check if it's impossible to reach 4 wins
                if wins < m + 1:
                    print(f"Stopping early: Variable cannot win 4 out of 5 games (wins: {wins}, games played: {m + 2})")
                    break
            else:
                # Loop completed without breaking, check if variable won 4 or more
                if wins >= 4:
                    # Variable won at least 4 out of 5 games, becomes new control
                    state_dict = {k: v.to('cpu') for k, v in variable_model.state_dict().items()}
                    control_model = AutoModelForCausalLM.from_pretrained(
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        quantization_config=quant_config
                    )
                    control_model.load_state_dict(state_dict)
                    # No need to move to CUDA; it's already on the GPU
                    # Removed: if torch.cuda.is_available():
                    #              control_model = control_model.to('cuda')
                    # Save the winning model
                    folder = f"New-LLM-Winner-{winner_count:02d}"
                    os.makedirs(folder, exist_ok=True)
                    torch.save(state_dict, f"{folder}/model.pt")
                    print(f"NEW WINNER!!! saved in this folder: {folder}")
                    winner_count += 1
        # If variable loses/ties first game or doesn't win 4/5, discard it and loop continues

if __name__ == "__main__":
    main()