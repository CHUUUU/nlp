import preprocessing.custom_dataset as custom_dataset
import time

if __name__ == "__main__":
    start = time.time()
    dataset = custom_dataset.Custom_dataset()
    train_data = dataset.get_data()
    print("time data load : ", time.time() - start)

    print(train_data[2:4])
