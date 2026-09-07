
import preprocessing
import features 



def main():
    print("Hello from vballPredictor!")
    preprocessor = preprocessing.DataPreprocessor()
    preprocessor.run()

    feature_engineer = features.FeatureEngineer()
    feature_engineer.run()




if __name__ == "__main__":
    main()
