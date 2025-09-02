import os
import asyncio
import ast
import pandas as pd
import matplotlib.pyplot as plt
from agents.random_agent import RandomAgent
from agents.agent_metric import AgentDQLCustomReward
from agents.agent_dqn import DQNAgent
from agents.larger_value import LargerValue
from rooms.room import Room
from datetime import datetime


def run_room_with_pretrained_agents(
    model_path_dql: str,
    model_path_attack: str,
    save_room_log: bool,
    save_game_dataset: bool,
    matches: int,
    output_folder: str,
):
    """
    Run a room with both pre-trained DQN agents loaded from their respective model paths
    """
    room = Room(
        run_remote_room=False,
        room_name="Room_DQL_vs_Attack_vs_Others",
        max_matches=matches,
        output_folder=output_folder,
        save_game_dataset=save_game_dataset,
        save_logs_game=save_room_log,
        save_logs_room=False,
    )

    # Create 1 random agent
    agent_random = RandomAgent(name="Random", log_directory=room.room_dir, verbose_log=False)
    room.connect_player(agent_random)

    # Create regular DQN agent with its model
    agent_dql = DQNAgent(
        "DQL",
        train=False,  # Set to False for testing only
        log_directory=room.room_dir,
        verbose_console=False,
        model_path=model_path_dql,
        load_model=True,  # Load the pre-trained model
    )
    room.connect_player(agent_dql)

    # Create attack DQN agent with its model
    agent_attack = AgentDQLCustomReward(
        "DQLCustomReward",
        train=False,  # Set to False for testing only
        log_directory=room.room_dir,
        verbose_console=False,
        model_path=model_path_attack,
        load_model=True,  # Load the pre-trained model
    )
    room.connect_player(agent_attack)

    # Create LargerValue agent
    agent_larger_value = LargerValue(
        "LargerValue",
        log_directory=room.room_dir,
        verbose_console=False
    )
    room.connect_player(agent_larger_value)
    
    asyncio.run(room.run())

    # Sanity checks for test mode
    print(f"[TEST] DQL Agent epsilon: {agent_dql.epsilon}")
    assert agent_dql.epsilon == 0.0, "DQL Epsilon should be 0 in test mode!"
    print(f"[TEST] DQL loaded model from: {agent_dql.model_path}")
    
    print(f"[TEST] Attack Agent epsilon: {agent_attack.epsilon}")
    assert agent_attack.epsilon == 0.0, "Attack Epsilon should be 0 in test mode!"
    print(f"[TEST] Attack loaded model from: {agent_attack.model_path}")
    
    return room, agent_dql, agent_attack


def plot_score_distribution(dataset_path: str, output_path: str):
    """Plot score distribution from the game dataset"""
    try:
        df = pd.read_csv(dataset_path, index_col=0)
        df = df[df["Action_Type"] == "END_MATCH"]
        names = ast.literal_eval(df.iloc[0]["Match_Score"])
        scores = df["Game_Score"].apply(ast.literal_eval).tolist()
        scores_arr = pd.DataFrame(scores, columns=names)
        
        plt.figure(figsize=(10, 6))
        for n in names:
            plt.plot(scores_arr[n], label=n, marker='o', markersize=2)
        plt.xlabel("Match")
        plt.ylabel("Score")
        plt.title("Score Progression - Classic DQN vs Random Agents")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Score progression plot saved to: {output_path}")
    except Exception as e:
        print(f"Error creating score plot: {e}")


if __name__ == "__main__":
    # Paths to the pre-trained models
    model_file_dql = os.path.join("src", "agents", "agent_dql_", "dql_modelvsAll.h5")
    model_file_attack = os.path.join("src", "agents", "agent_dql_att", "attack_agent_model.h5")
    
    # Check if both models exist
    models_to_check = [
        ("Regular DQL", model_file_dql),
        ("Attack DQL", model_file_attack)
    ]
    
    missing_models = []
    for model_name, model_path in models_to_check:
        if not os.path.exists(model_path):
            missing_models.append((model_name, model_path))
    
    if missing_models:
        print("ERROR: Missing model files:")
        for model_name, model_path in missing_models:
            print(f"  - {model_name}: {model_path}")
        
        # List available files in directories
        directories_to_check = [
            ("src/agents/agent_dql_/", os.path.join("src", "agents", "agent_dql_")),
            ("src/agents/agent_dql_att/", os.path.join("src", "agents", "agent_dql_att"))
        ]
        
        for dir_name, dir_path in directories_to_check:
            if os.path.exists(dir_path):
                print(f"\nAvailable files in {dir_name}:")
                for file in os.listdir(dir_path):
                    if file.endswith(('.h5', '.keras', '.pkl')):
                        print(f"  - {file}")
            else:
                print(f"\nDirectory {dir_name} does not exist")
        exit(1)
    
    print(f"Testing DQL vs Attack DQL vs LargerValue vs Random")
    print(f"Regular DQL model: {model_file_dql}")
    print(f"Attack DQL model: {model_file_attack}")
    now = datetime.now()
    
    # Run test matches
    test_room, test_agent_dql, test_agent_attack = run_room_with_pretrained_agents(
        model_path_dql=model_file_dql,
        model_path_attack=model_file_attack,
        save_room_log=False,
        save_game_dataset=False,
        matches=100,
        output_folder="outputs_test"
    )
    
    # Generate plots for regular DQL agent
    test_agent_dql.plot_score_progression(
        os.path.join(test_room.room_dir, "dql_score_progression.png")
    )
    test_agent_dql.plot_positions(
        os.path.join(test_room.room_dir, "dql_positions.png")
    )
    
    # Generate plots for attack DQL agent
    test_agent_attack.plot_score_progression(
        os.path.join(test_room.room_dir, "attack_score_progression.png")
    )
    test_agent_attack.plot_positions(
        os.path.join(test_room.room_dir, "attack_positions.png")
    )
    
    # Plot score distribution from dataset if available
    dataset_file = os.path.join(test_room.room_dir, "dataset", "game_dataset.pkl.csv")
    if os.path.exists(dataset_file):
        plot_score_distribution(
            dataset_file, 
            os.path.join(test_room.room_dir, "detailed_score_progression.png")
        )
    
    print(f"TESTING COMPLETED! Testing time: {(datetime.now() - now).total_seconds():.2f} seconds")
    print(f"Results saved in: {test_room.room_dir}")
    
    # Print statistics for both agents
    print(f"\n=== PERFORMANCE COMPARISON ===")
    
    if hasattr(test_agent_dql, 'positions') and test_agent_dql.positions:
        avg_position_dql = sum(test_agent_dql.positions) / len(test_agent_dql.positions)
        wins_dql = test_agent_dql.positions.count(1)
        print(f"\nRegular DQL Performance:")
        print(f"  Average position: {avg_position_dql:.2f}")
        print(f"  Wins (1st place): {wins_dql}/{len(test_agent_dql.positions)} ({wins_dql/len(test_agent_dql.positions)*100:.1f}%)")
        print(f"  Position distribution: {[test_agent_dql.positions.count(i) for i in range(1, 5)]}")
    
    if hasattr(test_agent_attack, 'positions') and test_agent_attack.positions:
        avg_position_attack = sum(test_agent_attack.positions) / len(test_agent_attack.positions)
        wins_attack = test_agent_attack.positions.count(1)
        print(f"\nAttack DQL Performance:")
        print(f"  Average position: {avg_position_attack:.2f}")
        print(f"  Wins (1st place): {wins_attack}/{len(test_agent_attack.positions)} ({wins_attack/len(test_agent_attack.positions)*100:.1f}%)")
        print(f"  Position distribution: {[test_agent_attack.positions.count(i) for i in range(1, 5)]}")
    
    # Compare performance
    if hasattr(test_agent_dql, 'positions') and hasattr(test_agent_attack, 'positions'):
        print(f"\n=== COMPARISON ===")
        if avg_position_dql < avg_position_attack:
            print(f"Regular DQL performed better (lower average position: {avg_position_dql:.2f} vs {avg_position_attack:.2f})")
        elif avg_position_attack < avg_position_dql:
            print(f"Attack DQL performed better (lower average position: {avg_position_attack:.2f} vs {avg_position_dql:.2f})")
        else:
            print(f"Both agents performed equally (same average position: {avg_position_dql:.2f})")
        
        print(f"Win rate comparison: Regular DQL {wins_dql/len(test_agent_dql.positions)*100:.1f}% vs Attack DQL {wins_attack/len(test_agent_attack.positions)*100:.1f}%")
