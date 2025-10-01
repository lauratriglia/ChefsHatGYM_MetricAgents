
import os
import asyncio
import ast
import pandas as pd
import matplotlib.pyplot as plt
from agents.random_agent import RandomAgent
from agents.agent_dqn import DQNAgent
from agents.agent_metric import AgentDQLCustomReward
from agents.agent_dqn_31 import AgentDQN31
from rooms.room import Room
from datetime import datetime


def run_room(
    training: bool,
    model_path: str,
    save_room_log: bool,
    save_game_dataset: bool,
    matches: int,
    output_folder: str,
):
    room = Room(
        run_remote_room=False,
        room_name="Room_DQL_atk_vs_Random_1kmatches_",
        max_matches=matches,
        output_folder=output_folder,
        save_game_dataset=save_game_dataset,
        save_logs_game=save_room_log,
        save_logs_room=False,
    )

    # Connect 3 random agents
    agents = [
        RandomAgent(name=f"Random{i}", log_directory=room.room_dir, verbose_log=False)
        for i in range(3)
    ]
    for a in agents:
        room.connect_player(a)
    reward = "attack"
    # Connect the DQL agent with custom reward
    # agent = AgentDQLCustomReward(
    #     f"DQL_{reward}",
    #     train=training,
    #     log_directory=room.room_dir,
    #     verbose_console=False,
    #     model_path=model_path,
    #     load_model=not training,
    #     reward_type=reward
    # )
    agent = DQNAgent(
        f"DQL_",
        train=training,
        log_directory=room.room_dir,
        verbose_console=False,
        model_path=model_path,
        load_model=not training,
    )
    # agent = AgentDQN31(
    #     f"DQL_31",
    #     train=training,
    #     log_directory=room.room_dir,
    #     verbose_console=False,
    #     model_path=model_path,
    #     load_model=not training,
    # )
    room.connect_player(agent)
    asyncio.run(room.run())

    # --- Test mode sanity checks ---
    if not training:
        print(f"[TEST] Epsilon: {agent.epsilon}")
        assert agent.epsilon == 0.0, "Epsilon should be 0 in test mode!"
        if hasattr(agent, "model_path"):
            print(f"[TEST] Loaded policy model: {agent.model_path}")
            target_path = agent.model_path.replace(".h5", ".target.h5")
            print(f"[TEST] Loaded target model: {target_path}")

    return room, agent


def plot_score_distribution(dataset_path: str, output_path: str):
    df = pd.read_csv(dataset_path, index_col=0)
    df = df[df["Action_Type"] == "END_MATCH"]
    names = ast.literal_eval(df.iloc[0]["Match_Score"])
    scores = df["Game_Score"].apply(ast.literal_eval).tolist()
    scores_arr = pd.DataFrame(scores, columns=names)
    plt.figure()
    for n in names:
        plt.plot(scores_arr[n], label=n)
    plt.xlabel("Match")
    plt.ylabel("Score")
    plt.title("Score Progression")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    model_file = os.path.join("outputs", "dql_model.h5")
    
    now = datetime.now()
    train_room, train_agent = run_room(
        True,
        model_file,
        False,
        False,
        10,
        "outputs",
    )
    
    # Plot training results
    train_agent.plot_loss(os.path.join(train_room.room_dir, "training_loss.png"))
    train_agent.plot_positions(os.path.join(train_room.room_dir, "training_positions.png"))
    train_agent.plot_score_progression(os.path.join(train_room.room_dir, "training_progression.png"))
    train_agent.plot_rewards(os.path.join(train_room.room_dir, "training_rewards_positions.png"))
    
    
    print(f"TRAINING DONE! Training time: {(datetime.now() - now).total_seconds()}")
    now = datetime.now()

    # # === TEST ===
    # test_room, test_agent = run_room(
    #     False, model_file, False, True, 100, "outputs_test"
    # )
    
    # # Plot test results
    # test_agent.plot_score_progression(os.path.join(test_room.room_dir, "test_progression.png"))
    # test_agent.plot_positions(os.path.join(test_room.room_dir, "test_positions.png"))
    
    # print(f"TESTING DONE! Testing time: {(datetime.now() - now).total_seconds()}")

    
