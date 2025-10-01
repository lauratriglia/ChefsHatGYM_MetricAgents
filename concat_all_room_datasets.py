import os
import pandas as pd

def concat_all_room_datasets(base_dir="outputs_test/DQL_Order/"):
    # Find all Room_DQL_OrderTest_* folders
    room_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("Room_DQL_OrderTest")]
    all_dfs = []
    for room_dir in room_dirs:
        dataset_dir = os.path.join(room_dir, "dataset")
        if not os.path.isdir(dataset_dir):
            print(f"No dataset folder in {room_dir}")
            continue
        for fname in os.listdir(dataset_dir):
            if fname.endswith(".csv"):
                fpath = os.path.join(dataset_dir, fname)
                try:
                    df = pd.read_csv(fpath)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
    print(all_dfs)                
    if all_dfs:
        concat_df = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(base_dir, "all_rooms_concatenated.pkl")
        concat_df.to_pickle(out_path)
        print(f"Concatenated dataset saved to {out_path}")
    else:
        print("No datasets found to concatenate.")

if __name__ == "__main__":
    concat_all_room_datasets()
