from Dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset("heart.csv", 5)
    dataset.initialize_the_model()
    dataset.train_model(1000)
    dataset.predict()
