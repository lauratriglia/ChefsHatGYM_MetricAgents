import itertools
import os
import asyncio
import itertools
import pandas as pd
import glob
from agents.random_agent import RandomAgent
from agents.agent_dqn_31 import AgentDQN31
from agents.agent_metric import AgentDQLCustomReward
from agents.larger_value import LargerValue
from rooms.room import Room
from datetime import datetime


import matplotlib.pyplot as plt

def run_room_with_order(agent_order, training, model_path_dql, model_path_attack, output_folder):
    room = Room(
        run_remote_room=False,
        room_name="Room_DQL_OrderTest",
        max_matches=10,
        output_folder=output_folder,
        save_game_dataset=True,
        save_logs_game=False,
        save_logs_room=False,
    )
    agent_constructors = [
        lambda room: RandomAgent(name="Random", log_directory=room.room_dir, verbose_log=False),
        lambda room: AgentDQN31("DQL_", train=training, log_directory=room.room_dir, verbose_console=False, model_path=model_path_dql, load_model=not training),
        lambda room: AgentDQLCustomReward("DQL_attack", train=training, log_directory=room.room_dir, verbose_console=False, model_path=model_path_attack, load_model=not training, reward_type="attack"),
        lambda room: LargerValue("LargerValue", log_directory=room.room_dir, verbose_console=False)
    ]
    agent_names = ["Random", "DQL_", "DQL_attack", "LargerValue"]
    agents = [agent_constructors[i](room) for i in agent_order]
    print(f"Testing order: {[agent_names[i] for i in agent_order]}")
    for agent in agents:
        room.connect_player(agent)
    asyncio.run(room.run())

    # Plot score progression for each agent in 4 subplots
    plt.figure(figsize=(12, 8))
    for idx, agent in enumerate(agents):
        plt.subplot(2, 2, idx+1)
        if hasattr(agent, "score_history") and agent.score_history:
            # score_history is a list of dicts: {player: score}
            scores = [score.get(agent.name, None) for score in agent.score_history]
            plt.plot(scores, marker='o')
            plt.title(f"{agent.name} Score Progression")
            plt.xlabel("Match")
            plt.ylabel("Score")
        else:
            plt.title(f"{agent.name} (no score data)")
    plt.tight_layout()
    order_str = "_".join([agent.name for agent in agents])
    plt.savefig(os.path.join(room.room_dir, f"score_progression_{order_str}.png"))
    plt.close()
    
    return room.room_dir  # Return room directory for dataset concatenation

def concatenate_all_datasets(output_base_folder, final_output_path):
    """
    Concatenate all datasets from different permutations into a single dataset
    """
    print("Starting dataset concatenation...")
    
    # Find all dataset files in the output folders
    dataset_pattern = os.path.join(output_base_folder, "**/dataset/game_dataset.pkl.csv")
    dataset_files = glob.glob(dataset_pattern, recursive=True)
    
    if not dataset_files:
        print("No dataset files found for concatenation.")
        return
    
    print(f"Found {len(dataset_files)} dataset files to concatenate:")
    for file in dataset_files:
        print(f"  - {file}")
    
    all_dataframes = []
    
    for i, dataset_file in enumerate(dataset_files):
        try:
            # Read the dataset
            df = pd.read_csv(dataset_file)
            
            # Add columns to identify the permutation
            room_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_file)))
            df['Permutation_ID'] = i
           
            all_dataframes.append(df)
            print(f"Added dataset {i+1}/{len(dataset_files)}: {len(df)} rows from {room_name}")
            
        except Exception as e:
            print(f"Error reading {dataset_file}: {e}")
            continue
    
    if not all_dataframes:
        print("No valid datasets found for concatenation.")
        return
    
    # Concatenate all dataframes
    final_dataset = pd.concat(all_dataframes, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    # Save the concatenated dataset
    final_dataset.to_csv(final_output_path, index=False)
    
    print(f"\nDataset concatenation completed!")
    print(f"Final dataset saved to: {final_output_path}")
    print(f"Total rows: {len(final_dataset)}")
    print(f"Columns: {list(final_dataset.columns)}")
    
    return final_dataset

def plot_average_scores_from_dataset(concatenated_dataset, output_path):
    """
    Plot average scores for the four players from the concatenated dataset
    """
    print("\nCreating average scores plot from concatenated dataset...")
    
    try:
        # Filter for END_MATCH entries which contain the final scores
        end_match_data = concatenated_dataset[concatenated_dataset['Action_Type'] == 'END_MATCH'].copy()
        
        if end_match_data.empty:
            print("No END_MATCH data found in concatenated dataset.")
            return
        
        # Parse Game_Score column (should contain score dictionaries)
        import ast
        
        all_scores = []
        agent_names = set()
        
        for idx, row in end_match_data.iterrows():
            try:
                # Parse the game score (should be a string representation of a dict)
                game_scores = ast.literal_eval(row['Game_Score']) if isinstance(row['Game_Score'], str) else row['Game_Score']
                
                if isinstance(game_scores, dict):
                    # Add permutation info to each score entry
                    score_entry = game_scores.copy()
                    score_entry['Permutation_ID'] = row['Permutation_ID']
                    score_entry['Match'] = row['Match']
                    all_scores.append(score_entry)
                    agent_names.update(game_scores.keys())
                    
            except Exception as e:
                print(f"Error parsing scores for row {idx}: {e}")
                continue
        
        if not all_scores:
            print("No valid score data found.")
            return
        
        # Convert to DataFrame for easier manipulation
        scores_df = pd.DataFrame(all_scores)
        agent_names = sorted([name for name in agent_names if name not in ['Permutation_ID', 'Match']])
        
        print(f"Found agents: {agent_names}")
        print(f"Total matches processed: {len(scores_df)}")
        
        # Calculate average scores for each agent
        avg_scores = {}
        std_scores = {}
        
        for agent in agent_names:
            if agent in scores_df.columns:
                agent_scores = pd.to_numeric(scores_df[agent], errors='coerce').dropna()
                avg_scores[agent] = agent_scores.mean()
                std_scores[agent] = agent_scores.std()
                print(f"{agent}: avg={avg_scores[agent]:.2f}, std={std_scores[agent]:.2f}, n={len(agent_scores)}")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Bar chart of average scores with error bars
        plt.subplot(2, 2, 1)
        agents = list(avg_scores.keys())
        avgs = list(avg_scores.values())
        stds = [std_scores[agent] for agent in agents]
        
        bars = plt.bar(agents, avgs, yerr=stds, capsize=5, alpha=0.7, 
                      color=['blue', 'red', 'green', 'orange'][:len(agents)])
        plt.title('Average Scores Across All Permutations')
        plt.ylabel('Average Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, avg in zip(bars, avgs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{avg:.1f}', ha='center', va='bottom')
        
        # Plot 2: Box plot of score distributions
        plt.subplot(2, 2, 2)
        score_data = [pd.to_numeric(scores_df[agent], errors='coerce').dropna().values 
                     for agent in agents if agent in scores_df.columns]
        plt.boxplot(score_data, labels=agents)
        plt.title('Score Distribution by Agent')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        # Plot 3: Score progression over matches (sample of permutations)
        plt.subplot(2, 2, 3)
        sample_permutations = scores_df['Permutation_ID'].unique()[:5]  # Show first 5 permutations
        
        for perm_id in sample_permutations:
            perm_data = scores_df[scores_df['Permutation_ID'] == perm_id].sort_values('Match')
            for agent in agents:
                if agent in perm_data.columns:
                    agent_scores = pd.to_numeric(perm_data[agent], errors='coerce')
                    plt.plot(perm_data['Match'], agent_scores, 
                            label=f'{agent}_P{perm_id}' if perm_id == sample_permutations[0] else "", 
                            alpha=0.6, linewidth=1)
        
        plt.title('Score Progression (Sample Permutations)')
        plt.xlabel('Match Number')
        plt.ylabel('Score')
        if sample_permutations[0] == sample_permutations[0]:  # Show legend only once
            plt.legend()
        
        # Plot 4: Average scores by permutation
        plt.subplot(2, 2, 4)
        perm_avgs = scores_df.groupby('Permutation_ID')[agents].mean()
        
        for agent in agents:
            if agent in perm_avgs.columns:
                plt.plot(perm_avgs.index, perm_avgs[agent], 'o-', label=agent, alpha=0.7)
        
        plt.title('Average Scores by Permutation')
        plt.xlabel('Permutation ID')
        plt.ylabel('Average Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Average scores plot saved to: {output_path}")
        
        # Save detailed statistics to text file
        stats_path = output_path.replace('.png', '_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("AVERAGE SCORES ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total matches analyzed: {len(scores_df)}\n")
            f.write(f"Total permutations: {scores_df['Permutation_ID'].nunique()}\n\n")
            
            f.write("AGENT PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for agent in agents:
                if agent in avg_scores:
                    f.write(f"{agent:12}: avg={avg_scores[agent]:6.2f}, std={std_scores[agent]:5.2f}\n")
            
            f.write("\nRANKING (by average score):\n")
            f.write("-" * 30 + "\n")
            sorted_agents = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (agent, score) in enumerate(sorted_agents, 1):
                f.write(f"{rank}. {agent}: {score:.2f}\n")
        
        print(f"Detailed statistics saved to: {stats_path}")
        
    except Exception as e:
        print(f"Error creating average scores plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model_file_dql = os.path.join("src/agents/agent_dql_31_", "dql31_model.h5")
    model_file_attack = os.path.join("src/agents/agent_metric", "dql_model.h5")
    
    now = datetime.now()
    output_base = "outputs_test/DQL_Order/"
    room_directories = []
    
    # Test all possible agent orders for 10 matches each
    print("Starting permutation tests...")
    for i, agent_order in enumerate(itertools.permutations(range(4))):
        print(f"\nRunning permutation {i+1}/24: {agent_order}")
        room_dir = run_room_with_order(agent_order, False, model_file_dql, model_file_attack, output_base)
        room_directories.append(room_dir)
    
    print(f"\nALL ORDER TESTS DONE! Total time: {(datetime.now() - now).total_seconds()}")
    
    # Concatenate all datasets into a single comprehensive dataset
    final_dataset_path = os.path.join(output_base, "concatenated_all_permutations_dataset_500.csv")
    print(f"\nStarting dataset concatenation...")
    
    concatenated_dataset = concatenate_all_datasets(output_base, final_dataset_path)
    
    # Plot average scores from the concatenated dataset
    if concatenated_dataset is not None:
        scores_plot_path = os.path.join(output_base, "average_scores_analysis.png")
        plot_average_scores_from_dataset(concatenated_dataset, scores_plot_path)
    
    print(f"\nTotal execution time: {(datetime.now() - now).total_seconds():.2f} seconds")




    
