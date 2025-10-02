import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def extract_unique_plays(training_csv_path: str):
    """
    Extract all unique plays from the training_data.csv document column.
    Each document is a JSON with play_id, play details, and context.
    """
    print("Loading training data...")
    df = pd.read_csv(training_csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Unique queries: {df['query'].nunique()}")
    print(f"Unique inspiration_play_ids: {df['inspiration_play_id'].nunique()}")
    
    unique_plays = {}
    play_id_counter = defaultdict(int)
    
    print("\nExtracting unique plays from 'document' column...")
    for idx, row in df.iterrows():
        doc_json = json.loads(row['document'])
        play_id = doc_json['play_id']
        
        play_id_counter[play_id] += 1
        
        if play_id not in unique_plays:
            unique_plays[play_id] = {
                'play_id': play_id,
                'play': doc_json['play'],
                'context': doc_json['context']
            }
    
    print(f"\nTotal unique plays extracted: {len(unique_plays)}")
    print(f"Plays with multiple references: {sum(1 for count in play_id_counter.values() if count > 1)}")
    print(f"Most referenced play: {max(play_id_counter.values())} times")
    
    return list(unique_plays.values()), play_id_counter

def create_play_text_representation(play_data: dict) -> str:
    """
    Create rich text representation for embedding.
    Includes teams, situation context, play history, play details, and description.
    """
    play = play_data['play']
    context = play_data['context']
    situation = context['situation']
    history = context.get('history', [])
    
    formation = play.get('formation', 'UNKNOWN')
    playtype = play.get('playtype', 'UNKNOWN')
    direction = play.get('direction', 'NONE')
    yards = play.get('yards', '0')
    description = play.get('description', '')
    
    yardline = situation['yardline']
    down = situation['down']
    togo = situation['togo']
    quarter = situation['quarter']
    offense = situation.get('offense', '')
    defense = situation.get('defense', '')
    
    text = ""
    
    if offense and defense:
        text += f"{offense} offense vs {defense} defense. "
    
    text += f"Situation: {down} and {togo} at {yardline} yard line, quarter {quarter}. "
    
    if history and len(history) > 0:
        history_text = []
        for h in history[:5]:
            h_type = h.get('playtype', '')
            h_dir = h.get('direction', '')
            h_yards = h.get('yards', '0')
            if h_type:
                h_str = h_type
                if h_dir and h_dir != 'null':
                    h_str += f" {h_dir}"
                h_str += f" {h_yards}yd"
                history_text.append(h_str)
        if history_text:
            text += f"Recent plays: {', '.join(history_text)}. "
    
    if formation and formation != 'null':
        text += f"Formation: {formation}. "
    
    text += f"Play: {playtype}"
    if direction and direction != 'null' and direction != 'NONE':
        text += f" {direction}"
    text += f" for {yards} yards. "
    
    if description:
        text += f"Description: {description}"
    
    return text.strip()

def prepare_corpus(training_csv_path: str, output_dir: Path):
    """
    Main function to prepare the play corpus.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_plays, play_id_counter = extract_unique_plays(training_csv_path)
    
    print("\nCreating text representations...")
    corpus = []
    for play_data in unique_plays:
        text = create_play_text_representation(play_data)
        corpus.append({
            'play_id': play_data['play_id'],
            'text': text,
            'play_data': play_data,
            'reference_count': play_id_counter[play_data['play_id']]
        })
    
    corpus_path = output_dir / 'play_corpus.json'
    print(f"\nSaving corpus to {corpus_path}...")
    with open(corpus_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"✓ Saved {len(corpus)} plays")
    
    print("\nSample play:")
    sample = corpus[0]
    print(f"  Play ID: {sample['play_id']}")
    print(f"  Text: {sample['text'][:200]}...")
    print(f"  Referenced: {sample['reference_count']} times")
    
    stats = {
        'total_plays': len(corpus),
        'avg_text_length': sum(len(p['text']) for p in corpus) / len(corpus),
        'play_types': defaultdict(int),
        'formations': defaultdict(int)
    }
    
    for play in corpus:
        playtype = play['play_data']['play'].get('playtype', 'UNKNOWN')
        formation = play['play_data']['play'].get('formation', 'UNKNOWN')
        stats['play_types'][playtype] += 1
        stats['formations'][formation] += 1
    
    stats['play_types'] = dict(stats['play_types'])
    stats['formations'] = dict(stats['formations'])
    
    stats_path = output_dir / 'corpus_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved statistics to {stats_path}")
    print(f"\nPlay type distribution:")
    for playtype, count in sorted(stats['play_types'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {playtype}: {count} ({100*count/stats['total_plays']:.1f}%)")
    
    return corpus

if __name__ == "__main__":
    training_csv = Path(__file__).parent.parent.parent / "training_data.csv"
    output_dir = Path(__file__).parent.parent / "DATA"
    
    print("=" * 70)
    print("NFL Play Corpus Preparation")
    print("=" * 70)
    
    corpus = prepare_corpus(training_csv, output_dir)
    
    print("\n" + "=" * 70)
    print("Corpus preparation complete!")
    print("=" * 70)
