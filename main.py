import tensorflow as tf
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import create_model, train_model, evaluate_model
from src.evaluate import plot_loss_curves, plot_lr_vs_loss, evaluate_predictions

def main():
    try:
        file_path = 'src/dataset/employee_attrition.csv'
        df = load_data(file_path)
        if df is None:
            return

        x_train, x_test, y_train, y_test = preprocess_data(df)
        if x_train is None:
            return

        # Model 1: Simple model
        model_1 = create_model([1])
        if model_1 is None:
            return
        train_model(model_1, x_train, y_train, epochs=5)
        evaluate_model(model_1, x_train, y_train)

        # Model 2: Train for longer
        history_2 = train_model(model_1, x_train, y_train, epochs=100)
        if history_2 is None:
            return
        evaluate_model(model_1, x_train, y_train)
        plot_loss_curves(history_2, "model_2_training_curves.png")

        # Model 3: Extra layer
        model_3 = create_model([1, 1])
        if model_3 is None:
            return
        history_3 = train_model(model_3, x_train, y_train, epochs=50)
        if history_3 is None:
            return
        evaluate_model(model_3, x_train, y_train)
        plot_loss_curves(history_3, "model_3_training_curves.png")

        # Model 4: 2 neurons in the first layer
        model_4 = create_model([2, 1])
        if model_4 is None:
            return
        history_4 = train_model(model_4, x_train, y_train, epochs=50)
        if history_4 is None:
            return
        evaluate_model(model_4, x_train, y_train)
        plot_loss_curves(history_4, "model_4_training_curves.png")

        # Model 5: 3 layers
        model_5 = create_model([1, 1, 1])
        if model_5 is None:
            return
        history_5 = train_model(model_5, x_train, y_train, epochs=50)
        if history_5 is None:
            return
        evaluate_model(model_5, x_train, y_train)
        plot_loss_curves(history_5, "model_5_training_curves.png")

        # Model 6: Learning rate 0.0009
        model_6 = create_model([1, 1], learning_rate=0.0009)
        if model_6 is None:
            return
        history_6 = train_model(model_6, x_train, y_train, epochs=50)
        if history_6 is None:
            return
        evaluate_model(model_6, x_train, y_train)
        plot_loss_curves(history_6, "model_6_training_curves.png")

        # Model 7: Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.9**(epoch/3))
        history_7 = train_model(model_6, x_train, y_train, epochs=100, callbacks=[lr_scheduler])
        if history_7 is None:
            return
        plot_lr_vs_loss(history_7, "model_7_lr_vs_loss.png")

        # Model 8: Activation function
        model_8 = create_model([1, 1], learning_rate=0.0009, activation='sigmoid')
        if model_8 is None:
            return
        history_8 = train_model(model_8, x_train, y_train, epochs=50)
        if history_8 is None:
            return
        evaluate_model(model_8, x_train, y_train)
        plot_loss_curves(history_8, "model_8_training_curves.png")

        # Evaluate predictions
        evaluate_predictions(model_8, x_test, y_test)

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
