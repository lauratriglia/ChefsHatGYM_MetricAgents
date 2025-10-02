import itertools
import os
import asyncio
import itertools
import pandas as pd
import glob
import ast
from agents.random_agent import RandomAgent
from agents.agent_dqn_31 import AgentDQN31
from agents.agent_metric import AgentDQLCustomReward
from agents.larger_value import LargerValue
from rooms.room import Room
from datetime import datetime
import matplotlib.pyplot as plt

def plot_average_scores_from_dataset(concatenated_dataset, output_path):
    """
    Plot average scores for the four players from the concatenated dataset
    Working with columns: ['Match', 'Round', 'Agent_Names', 'Source', 'Action_Type', 
    'Action_Description', 'Player_Finished', 'Player_Hands', 'Board_Before', 
    'Board_After', 'Possible_Actions', 'Current_Roles', 'Match_Score', 'Game_Score', 
    'Game_Performance_Score']
    """
    print("\nCreating average scores plot from concatenated dataset...")
    print(f"Dataset shape: {concatenated_dataset.shape}")
    print(f"Available columns: {list(concatenated_dataset.columns)}")
    
    try:
        # Filter for END_MATCH entries which contain the final scores
        end_match_data = concatenated_dataset[concatenated_dataset['Action_Type'] == 'END_MATCH'].copy()
        
        if end_match_data.empty:
            print("No END_MATCH data found in concatenated dataset.")
            return
        
        print(f"Found {len(end_match_data)} END_MATCH entries")
        
        all_scores = []
        agent_names = set()
        
        # Since we don't have explicit Permutation_ID, we'll estimate it based on match patterns
        # Assuming each permutation ran a similar number of matches
        matches_per_permutation = 100  # Adjust this based on your actual setup
        
        for idx, row in end_match_data.iterrows():
            try:
                # Parse the game score (should be a string representation of a dict)
                game_scores = ast.literal_eval(row['Game_Score']) if isinstance(row['Game_Score'], str) else row['Game_Score']
                
                if isinstance(game_scores, dict):
                    # Create synthetic Permutation_ID based on match number
                    # This assumes matches are numbered sequentially across permutations
                    synthetic_perm_id = row['Match'] // matches_per_permutation
                    
                    # Add match info to each score entry
                    score_entry = game_scores.copy()
                    score_entry['Permutation_ID'] = synthetic_perm_id
                    score_entry['Match'] = row['Match']
                    score_entry['Round'] = row['Round']
                    
                    # Try to extract agent names order if available
                    if 'Agent_Names' in row and pd.notna(row['Agent_Names']):
                        try:
                            agent_list = ast.literal_eval(row['Agent_Names']) if isinstance(row['Agent_Names'], str) else row['Agent_Names']
                            score_entry['Agent_Order'] = str(agent_list)
                        except:
                            pass
                    
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
        agent_names = sorted([name for name in agent_names if name not in ['Permutation_ID', 'Match', 'Round', 'Agent_Order']])
        
        print(f"Found agents: {agent_names}")
        print(f"Total matches processed: {len(scores_df)}")
        print(f"Estimated permutations: {scores_df['Permutation_ID'].nunique()}")
        
        # Calculate average scores for each agent
        avg_scores = {}
        std_scores = {}
        
        for agent in agent_names:
            if agent in scores_df.columns:
                agent_scores = pd.to_numeric(scores_df[agent], errors='coerce').dropna()
                if len(agent_scores) > 0:
                    avg_scores[agent] = agent_scores.mean()
                    std_scores[agent] = agent_scores.std()
                    print(f"{agent}: avg={avg_scores[agent]:.2f}, std={std_scores[agent]:.2f}, n={len(agent_scores)}")
        
        if not avg_scores:
            print("No valid agent scores found for plotting.")
            return
        
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Bar chart of average scores with error bars
        agents = list(avg_scores.keys())
        avgs = list(avg_scores.values())
        stds = [std_scores[agent] for agent in agents]
        
        bars = axes[0,0].bar(agents, avgs, yerr=stds, capsize=5, alpha=0.7, 
                            color=['blue', 'red', 'green', 'orange', 'purple'][:len(agents)])
        axes[0,0].set_title('Average Scores Across All Matches')
        axes[0,0].set_ylabel('Average Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, avg in zip(bars, avgs):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                          f'{avg:.1f}', ha='center', va='bottom')
        
        # Plot 2: Box plot of score distributions
        score_data = [pd.to_numeric(scores_df[agent], errors='coerce').dropna().values 
                     for agent in agents if agent in scores_df.columns]
        box_plot = axes[0,1].boxplot(score_data, labels=agents, patch_artist=True)
        axes[0,1].set_title('Score Distribution by Agent')
        axes[0,1].set_ylabel('Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'plum']
        for patch, color in zip(box_plot['boxes'], colors[:len(agents)]):
            patch.set_facecolor(color)
        
        # Plot 3: Score progression over matches
        axes[1,0].set_title('Score Progression Over Matches')
        axes[1,0].set_xlabel('Match Number')
        axes[1,0].set_ylabel('Score')
        
        # Sample some matches to avoid overcrowding
        unique_matches = sorted(scores_df['Match'].unique())
        sample_matches = unique_matches[::max(1, len(unique_matches)//50)]  # Sample every nth match
        sample_data = scores_df[scores_df['Match'].isin(sample_matches)]
        
        for agent in agents:
            if agent in sample_data.columns:
                # Group by match and take the first score (in case of duplicates)
                match_scores = sample_data.groupby('Match')[agent].first()
                axes[1,0].plot(match_scores.index, match_scores.values, 'o-', 
                              label=agent, alpha=0.7, linewidth=1, markersize=3)
        
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Average scores by estimated permutation or histogram
        if scores_df['Permutation_ID'].nunique() > 1:
            perm_avgs = scores_df.groupby('Permutation_ID')[agents].mean()
            
            for agent in agents:
                if agent in perm_avgs.columns:
                    axes[1,1].plot(perm_avgs.index, perm_avgs[agent], 'o-', 
                                  label=agent, alpha=0.7, linewidth=2, markersize=5)
            
            axes[1,1].set_title('Average Scores by Estimated Permutation')
            axes[1,1].set_xlabel('Estimated Permutation ID')
            axes[1,1].set_ylabel('Average Score')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        else:
            # If no clear permutations, show score frequency distribution
            for i, agent in enumerate(agents):
                agent_scores = pd.to_numeric(scores_df[agent], errors='coerce').dropna()
                axes[1,1].hist(agent_scores, alpha=0.6, label=agent, bins=15, 
                              color=colors[i % len(colors)])
            
            axes[1,1].set_title('Score Frequency Distribution')
            axes[1,1].set_xlabel('Score')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Average scores plot saved to: {output_path}")
        
        # Save detailed statistics to text file
        stats_path = output_path.replace('.png', '_stats500.txt')
        with open(stats_path, 'w') as f:
            f.write("AVERAGE SCORES ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total matches analyzed: {len(scores_df)}\n")
            f.write(f"Estimated permutations: {scores_df['Permutation_ID'].nunique()}\n")
            f.write(f"Matches per permutation estimate: {matches_per_permutation}\n\n")
            
            f.write("AGENT PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for agent in agents:
                if agent in avg_scores:
                    f.write(f"{agent:15}: avg={avg_scores[agent]:6.2f}, std={std_scores[agent]:5.2f}\n")
            
            f.write("\nRANKING (by average score):\n")
            f.write("-" * 30 + "\n")
            sorted_agents = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (agent, score) in enumerate(sorted_agents, 1):
                f.write(f"{rank}. {agent}: {score:.2f}\n")
            
            f.write(f"\nDATASET INFO:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Available columns: {list(concatenated_dataset.columns)}\n")
            f.write(f"Total dataset rows: {len(concatenated_dataset)}\n")
            f.write(f"END_MATCH entries: {len(end_match_data)}\n")
            f.write(f"Unique matches: {len(unique_matches)}\n")
            
            # Additional analysis
            if 'Agent_Names' in concatenated_dataset.columns:
                agent_orders = end_match_data['Agent_Names'].value_counts()
                f.write(f"\nAGENT ORDER FREQUENCY:\n")
                f.write("-" * 30 + "\n")
                for order, count in agent_orders.head(10).items():
                    f.write(f"{order}: {count} times\n")
        
        print(f"Detailed statistics saved to: {stats_path}")
        
    except Exception as e:
        print(f"Error creating average scores plot: {e}")
        import traceback
        traceback.print_exc()

def load_and_plot_existing_dataset(dataset_path, output_path, matches_per_permutation=100):
    """
    Load an existing concatenated dataset and create plots
    
    Args:
        dataset_path: Path to the concatenated CSV file
        output_path: Path where to save the plot
        matches_per_permutation: Number of matches per permutation (for estimating permutation ID)
    """
    try:
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        
        plot_average_scores_from_dataset(df, output_path)
        
    except Exception as e:
        print(f"Error loading or plotting dataset: {e}")
        import traceback
        traceback.print_exc()

def analyze_dataset_structure(dataset_path):
    """
    Analyze the structure of the concatenated dataset to understand the data better
    """
    try:
        df = pd.read_csv(dataset_path)
        
        print("DATASET STRUCTURE ANALYSIS")
        print("=" * 50)
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nAction types: {df['Action_Type'].value_counts().to_dict()}")
        print(f"Unique matches: {df['Match'].nunique()}")
        print(f"Unique rounds: {df['Round'].nunique()}")
        
        # Analyze END_MATCH entries
        end_match = df[df['Action_Type'] == 'END_MATCH']
        print(f"\nEND_MATCH entries: {len(end_match)}")
        
        if len(end_match) > 0:
            # Try to parse first Game_Score to see agent names
            first_score = end_match.iloc[0]['Game_Score']
            try:
                import ast
                parsed_score = ast.literal_eval(first_score)
                print(f"Sample game score structure: {parsed_score}")
                print(f"Detected agents: {list(parsed_score.keys())}")
            except:
                print(f"Could not parse Game_Score: {first_score}")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return None

if __name__ == "__main__":
    # Example usage with an existing dataset
    dataset_file = "outputs_test/DQL_Order/concatenated_all_permutations_dataset_500.csv"
    output_plot = "outputs_test/DQL_Order/scores_analysis_from_existing_500.png"
    
    if os.path.exists(dataset_file):
        # First analyze the dataset structure
        print("Analyzing dataset structure...")
        analyze_dataset_structure(dataset_file)
        
        print("\nCreating plots...")
        load_and_plot_existing_dataset(dataset_file, output_plot, matches_per_permutation=100)
    else:
        print(f"Dataset file not found: {dataset_file}")
        print("Please update the dataset_file path to point to your concatenated dataset.")