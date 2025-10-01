import itertools
import os
import asyncio
import itertools
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
        max_matches=1,
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

if __name__ == "__main__":
    model_file_dql = os.path.join("src/agents/agent_dql_31_", "dql31_model.h5")
    model_file_attack = os.path.join("src/agents/agent_dql_att", "dql_model.h5")
    now = datetime.now()
    # Test all possible agent orders for 1 match
    for agent_order in itertools.permutations(range(4)):
        run_room_with_order(agent_order, False, model_file_dql, model_file_attack, "outputs_test/DQL_Order/")
    print(f"ALL ORDER TESTS DONE! Total time: {(datetime.now() - now).total_seconds()}")




    
