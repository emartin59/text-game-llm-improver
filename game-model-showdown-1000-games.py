import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Define quantization configuration for 8-bit precision
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", padding_side='left')
original_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=quant_config
)
challenger_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=quant_config
)

# Load the state dictionaries from the files
original_model.load_state_dict(torch.load("modelOriginalTinyLlama.pt", map_location='cpu'))
challenger_model.load_state_dict(torch.load("modelChallengerLLM.pt", map_location='cpu'))

# Global variables
next_id = 4
direct_clones = {}

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
    words = response.split()
    commands = [
        ("mate", "ma", "m", lambda: bot['energy'] >= 75 and any(abs(other['y'] - bot['y']) + abs(other['x'] - bot['x']) == 1 for other in bots if other != bot)),
        ("up", "up", "u", lambda: True),
        ("down", "do", "d", lambda: True),
        ("left", "le", "l", lambda: True),
        ("right", "ri", "r", lambda: True),
        ("eat", "ea", "e", lambda: bot['seeds'] >= 10),
        ("plant", "pl", "p", lambda: bot['seeds'] > 0 and terrain[bot['y']][bot['x']] == '0'),
        ("talk:", "ta", "t", lambda: True)
    ]
    for word in reversed(words):
        for cmd, _, _, condition in commands:
            if cmd == "talk:" and word.startswith("talk:") and condition():
                return ("Talk", response[len("talk:"):].strip())
            elif word == cmd and condition():
                return cmd.capitalize()
    for word in reversed(words):
        for cmd, _, _, condition in commands:
            if cmd == "talk:" and word.startswith("talk:") and condition():
                return ("Talk", response[len("talk:"):].strip())
            elif cmd in word and condition():
                return cmd.capitalize()
    for word in reversed(words):
        if len(word) >= 2:
            two_chars = word[:2]
            for cmd, two_prefix, _, condition in commands:
                if two_chars == two_prefix and condition():
                    if cmd == "talk:": return ("Talk", response.strip())
                    return cmd.capitalize()
    for word in reversed(words):
        for cmd, two_prefix, _, condition in commands:
            if two_prefix in word and condition():
                if cmd == "talk:": return ("Talk", response.strip())
                return cmd.capitalize()
    for word in reversed(words):
        if len(word) > 0:
            first_letter = word[0]
            for cmd, _, letter, condition in commands:
                if first_letter == letter and condition():
                    if cmd == "talk:": return ("Talk", response.strip())
                    return cmd.capitalize()
    for word in reversed(words):
        for cmd, _, letter, condition in commands:
            if letter in word and condition():
                if cmd == "talk:": return ("Talk", response.strip())
                return cmd.capitalize()
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
    """Initialize the 9x9 terrain."""
    terrain = [['u' if y == 0 or y == 8 or x == 0 or x == 8 else '0' for x in range(9)] for y in range(9)]
    inner_positions = [(y, x) for y in range(1, 8) for x in range(1, 8)]
    random.shuffle(inner_positions)
    for i in range(20): terrain[inner_positions[i][0]][inner_positions[i][1]] = 'w'
    for i in range(20, 25): terrain[inner_positions[i][0]][inner_positions[i][1]] = 'r'
    available = [(y, x) for y in range(1, 8) for x in range(1, 8) if terrain[y][x] == '0']
    random.shuffle(available)
    bot_positions = available[:4] if len(available) >= 4 else [(1, 1), (1, 2), (7, 6), (7, 7)]
    return terrain, bot_positions

def is_occupied(y, x, bots):
    """Check if a position is occupied."""
    return any(bot['y'] == y and bot['x'] == x for bot in bots)

def get_prompt(bot, bots, terrain):
    """Generate the prompt for a bot."""
    view = [row[:] for row in terrain]
    for other_bot in bots:
        if other_bot == bot: view[other_bot['y']][other_bot['x']] = 'x'
        elif other_bot['is_talking']: view[other_bot['y']][other_bot['x']] = 't'
        else: view[other_bot['y']][other_bot['x']] = 'z'
    map_view = '\n'.join(' '.join(row) for row in view)
    return f"""
You have {bot['energy']} energy and {bot['seeds']} seeds. You are 'x' on the map:

{map_view}

'r' is rock (push if space behind is empty), 'z' is another bot, 't' is a talking bot, 'w' is wheat (move onto it for 10 seeds), '0' is empty, 'u' is a wall.

Goal: Be the first to have 3 direct clones via mating. Mating requires a bot ('z' or 't') next to you and 75 energy.

Commands:
- 'Mate': Clone if next to a bot (75 energy)
- 'Up', 'Down', 'Left', 'Right': Move
- 'Eat': 10 seeds -> 10 energy (1 energy, needs 10 seeds)
- 'Plant': Plant a seed on '0' (1 energy, 1 seed, grows to 'w' in 10 turns)
- 'Talk: [message]': Send a message (1 energy)

One command only.
"""

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
                    if (new_y, new_x) in planted: del planted[(new_y, new_x)]
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

def run_game(original_starts):
    """Run a single game and return the winner ('original', 'challenger', or 'tie')."""
    global next_id, direct_clones
    terrain, bot_positions = create_terrain()
    bots = [
        {'id': 0, 'y': bot_positions[0][0], 'x': bot_positions[0][1], 'energy': 100, 'seeds': 0, 'last_action': None, 'is_talking': False, 'message': None, 'llm_type': 'original' if original_starts else 'challenger'},
        {'id': 1, 'y': bot_positions[1][0], 'x': bot_positions[1][1], 'energy': 100, 'seeds': 0, 'last_action': None, 'is_talking': False, 'message': None, 'llm_type': 'original' if original_starts else 'challenger'},
        {'id': 2, 'y': bot_positions[2][0], 'x': bot_positions[2][1], 'energy': 100, 'seeds': 0, 'last_action': None, 'is_talking': False, 'message': None, 'llm_type': 'challenger' if original_starts else 'original'},
        {'id': 3, 'y': bot_positions[3][0], 'x': bot_positions[3][1], 'energy': 100, 'seeds': 0, 'last_action': None, 'is_talking': False, 'message': None, 'llm_type': 'challenger' if original_starts else 'original'}
    ]
    next_id = 4
    direct_clones = {0: 0, 1: 0, 2: 0, 3: 0}
    action_count = 0
    planted = {}
    original_created = 2
    challenger_created = 2

    while True:
        active_bots = [bot for bot in bots if bot['energy'] > 0]
        original_bots = [bot for bot in active_bots if bot['llm_type'] == 'original']
        challenger_bots = [bot for bot in active_bots if bot['llm_type'] == 'challenger']

        original_prompts = [get_prompt(bot, bots, terrain) for bot in original_bots]
        original_responses = tinyllama_query(original_prompts, original_model) if original_prompts else []
        for bot, response in zip(original_bots, original_responses):
            action = get_action_from_response(response, bot, terrain, bots)
            bot['last_action'] = action
            if action == "Mate" and bot['energy'] >= 75: original_created += 1
            process_bot_action(bot, action, terrain, bots, planted, action_count)

        challenger_prompts = [get_prompt(bot, bots, terrain) for bot in challenger_bots]
        challenger_responses = tinyllama_query(challenger_prompts, challenger_model) if challenger_prompts else []
        for bot, response in zip(challenger_bots, challenger_responses):
            action = get_action_from_response(response, bot, terrain, bots)
            bot['last_action'] = action
            if action == "Mate" and bot['energy'] >= 75: challenger_created += 1
            process_bot_action(bot, action, terrain, bots, planted, action_count)

        for (y, x), count in list(planted.items()):
            if terrain[y][x] == 'v' and not is_occupied(y, x, bots):
                if action_count - count >= 10: terrain[y][x] = 'w'

        bots = [bot for bot in bots if bot['energy'] > 0]
        game_ending = (direct_clones.get(0, 0) >= 3 or direct_clones.get(1, 0) >= 3 or 
                       direct_clones.get(2, 0) >= 3 or direct_clones.get(3, 0) >= 3 or 
                       len(bots) == 0 or next_id >= 24)

        if game_ending:
            if original_created > challenger_created: return 'original'
            elif challenger_created > original_created: return 'challenger'
            else: return 'tie'
        action_count += 1

def main():
    """Run 1,000 games and track statistics."""
    original_wins = 0
    challenger_wins = 0
    ties = 0
    original_starts_wins = 0
    challenger_starts_wins = 0

    for game in range(1000):
        original_starts = (game % 2 == 0)
        winner = run_game(original_starts)
        if winner == 'original':
            original_wins += 1
            if original_starts: original_starts_wins += 1
        elif winner == 'challenger':
            challenger_wins += 1
            if not original_starts: challenger_starts_wins += 1
        else:
            ties += 1
        print(f"Original TinyLlama {original_wins:04d}. Challenger LLM {challenger_wins:04d}. Games Played {game + 1:04d}. {'Original' if original_starts else 'Challenger'} started.")

    # Final statistics
    print("\nFinal Statistics:")
    print(f"Total Games Played: 1000")
    print(f"Original TinyLlama Wins: {original_wins} ({original_wins / 10:.1f}%)")
    print(f"Challenger LLM Wins: {challenger_wins} ({challenger_wins / 10:.1f}%)")
    print(f"Ties: {ties} ({ties / 10:.1f}%)")
    print(f"Original Wins When Starting: {original_starts_wins} out of 500 ({original_starts_wins / 5:.1f}%)")
    print(f"Challenger Wins When Starting: {challenger_starts_wins} out of 500 ({challenger_starts_wins / 5:.1f}%)")
    input("Press Enter to exit...")  # Keeps console open

if __name__ == "__main__":
    main()