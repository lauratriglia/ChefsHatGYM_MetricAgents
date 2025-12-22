import os
import asyncio
import ast
import pandas as pd
import matplotlib.pyplot as plt
from agents.random_agent import RandomAgent
from agents.agent_dqn import DQNAgent
from agents.agent_dqn_31 import AgentDQN31
from agents.agent_metric import AgentDQLCustomReward
from agents.agent_metricP import AgentDQLMetrics
from rooms.room import Room
from datetime import datetime
from agents.larger_value import LargerValue

# Code for running a training or testing room with a DQL metric agent in three different scenario (cond1, cond2, cond3)
# Cond1: 28 state vector (no adding of the card remaining)
# Cond2: 31 state vector (adding the card remaining)
# Cond3: 43 state vector (adding the card remaining + counting the cards remaining in the deck)
# Model saved in the training room folder and then loaded in the test room
def run_room(
    training: bool,
    model_path: str,
    save_room_log: bool,
    save_game_dataset: bool,
    matches: int,
    output_folder: str,
):
    if training:
        room_name = "Train_DQN"
    else:
        room_name = "Test_DQN"
    room = Room(
        run_remote_room=False,
        room_name=room_name,
        max_matches=matches,
        output_folder=output_folder,
        save_game_dataset=save_game_dataset,
        save_logs_game=save_room_log,
        save_logs_room=False,
    )

    agents = [
        RandomAgent(name=f"Random{i}", log_directory=room.room_dir, verbose_log=False)
        for i in range(3)
    ]
    for a in agents:
        room.connect_player(a)

    reward = "attack"
    if training:
        agent = AgentDQLCustomReward(
            f"DQL_{reward}",
            train=training,
            log_directory=room.room_dir,
            verbose_console=False,
            model_path=os.path.join(room.room_dir, "dqn_model.h5"),
            load_model=not training,
            reward_type=reward,
        )
    else:
        agent = AgentDQLCustomReward(
            f"DQL_{reward}",
            train=training,
            log_directory=room.room_dir,
            verbose_console=False,
            model_path=model_path,
            load_model=not training,
            reward_type=reward,
        )
    # if training:
    #     agent = AgentDQLMetrics(
    #         f"DQL_{reward}",
    #         train=training,
    #         log_directory=room.room_dir,
    #         verbose_console=False,
    #         model_path=os.path.join(room.room_dir, "dqn_model.h5"),
    #         load_model=not training,
    #         reward_type=reward,
    #     )
    # else:
    #     agent = AgentDQLMetrics(
    #         f"DQL_{reward}",
    #         train=training,
    #         log_directory=room.room_dir,
    #         verbose_console=False,
    #         model_path=model_path,
    #         load_model=not training,
    #         reward_type=reward,
    #     )
    
    # agent = DQNAgent(
    #     f"DQN",
    #     train=training,
    #     log_directory=room.room_dir,
    #     verbose_console=False,
    #     model_path=model_path if not training else os.path.join(room.room_dir, "dqn_model.h5"),
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
        # Optionally: agent.model.summary()

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
    model_file = os.path.join("src/agents/agent_metric/atk", "dql_model.h5")
    now = datetime.now()
    test_room, test_agent = run_room(
         False, model_file, False, True, 100, "outputs_test"
     )
    dataset_file = os.path.join(test_room.room_dir, "dataset", "game_dataset.pkl.csv")
    test_agent.plot_score_progression(
         os.path.join(test_room.room_dir, "score_progression.png")
    )
    
    print(f"TESTING DONE! Testing time: {(datetime.now() - now).total_seconds()}")

