import tensorflow as tf
from keras import callbacks, layers, models
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

model = models.Sequential(
    [

        layers.Input(shape=(28, 28, 1)),
        
        layers.RandomRotation(0.08),  
        layers.RandomZoom(0.08),  
        layers.RandomTranslation(0.08, 0.08),  
        
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint = callbacks.ModelCheckpoint(
    "digit_model.keras", save_best_only=True, monitor="val_accuracy"
)

print("Training the improved model with data augmentation...")
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print("Best model saved successfully as 'digit_model.keras'")