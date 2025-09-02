import os
import asyncio
import ast
import pandas as pd
import matplotlib.pyplot as plt
from agents.random_agent import RandomAgent
from agents.agent_dqn import DQNAgent
from agents.agent_metric import AgentDQLCustomReward
from agents.larger_value import LargerValue
from rooms.room import Room
from datetime import datetime


def run_room(
    training: bool,
    model_path_dql: str,
    model_path_attack: str,
    save_room_log: bool,
    save_game_dataset: bool,
    matches: int,
    output_folder: str,
):
    room = Room(
        run_remote_room=False,
        room_name="Room_DQL_Test_New_DQN",
        max_matches=matches,
        output_folder=output_folder,
        save_game_dataset=save_game_dataset,
        save_logs_game=save_room_log,
        save_logs_room=False,
    )

    #agents = [
    #    RandomAgent(name=f"Random{i}", log_directory=room.room_dir, verbose_log=False)
    #    for i in range(1)
    #]
    #for a in agents:
    #    room.connect_player(a)
    agent_random = RandomAgent(name=f"Random", log_directory=room.room_dir, verbose_log=False)
    room.connect_player(agent_random)   
    agent = DQNAgent(
        "DQL",
        train=training,
        log_directory=room.room_dir,
        verbose_console=False,
        model_path=model_path_dql,
        load_model=not training,
    )
    room.connect_player(agent)
    agent_dql_custom_reward = AgentDQLCustomReward(
        "DQLCustomReward",
        train=training,
        log_directory=room.room_dir,
        verbose_console=False,
        model_path=model_path_attack,
        load_model=not training,
    )
    
    room.connect_player(agent_dql_custom_reward)
    agent_larger_value = LargerValue(
        "LargerValue",
        log_directory=room.room_dir,
        verbose_console=False
    )
    room.connect_player(agent_larger_value)
    asyncio.run(room.run())

    # --- Test mode sanity checks ---
    if not training:
        print(f"[TEST] Epsilon: {agent.epsilon}")
        assert agent.epsilon == 0.0, "Epsilon should be 0 in test mode!"
        if hasattr(agent, "model_path"):
            print(f"[TEST] Loaded policy model: {agent.model_path}")
            target_path = agent.model_path.replace(".h5", ".target.h5")
            print(f"[TEST] Loaded target model: {target_path}")
        # Optionally: agent.model.summary()

    return room, agent, agent_dql_custom_reward


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
    model_file_dql = os.path.join("outputs", "dql_model.h5")
    model_file_attack = os.path.join("outputs", "attack_agent_model.h5")
    
    now = datetime.now()
    train_room, train_agent_dql, train_agent_attack = run_room(
        True,
        model_file_dql,
        model_file_attack,
        True,
        False,
        1000,
        "outputs",
    )
    
    # Plot results for regular DQL agent
    train_agent_dql.plot_loss(os.path.join(train_room.room_dir, "dql_training_loss.png"))
    train_agent_dql.plot_positions(os.path.join(train_room.room_dir, "dql_training_positions.png"))
    train_agent_dql.plot_score_progression(os.path.join(train_room.room_dir, "dql_training_progression.png"))
    
    # Plot results for attack agent
    train_agent_attack.plot_loss(os.path.join(train_room.room_dir, "attack_training_loss.png"))
    train_agent_attack.plot_positions(os.path.join(train_room.room_dir, "attack_training_positions.png"))
    train_agent_attack.plot_score_progression(os.path.join(train_room.room_dir, "attack_training_progression.png"))
    #train_agent_attack.plot_time(os.path.join(train_room.room_dir, "attack_training_time.png"))
    
    print(f"TRAINING DONE! Training time: {(datetime.now() - now).total_seconds()}")
    now = datetime.now()

    # === TEST ===
    test_room, test_agent_dql, test_agent_attack = run_room(
        False, model_file_dql, model_file_attack, False, True, 100, "outputs_test"
    )
    
    # Plot test results
    test_agent_dql.plot_score_progression(os.path.join(test_room.room_dir, "dql_test_progression.png"))
    test_agent_attack.plot_score_progression(os.path.join(test_room.room_dir, "attack_test_progression.png"))
    
    print(f"TESTING DONE! Testing time: {(datetime.now() - now).total_seconds()}")

    # Optionally plot score distribution if dataset exists
    # plot_score_distribution(
    #     dataset_file, os.path.join(test_room.room_dir, "score_progression.png")
    # )
