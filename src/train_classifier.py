import os
import tensorflow as tf
from tensorflow.keras import layers, models


DATA_DIR = "data/Train"        
IMG_SIZE = (224, 224)    
BATCH_SIZE = 32          
SEED = 42

def load_data():
    print("Loading Training Dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,      
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    print("Loading Validation Dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"\nFound Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(num_classes):

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape = IMG_SIZE + (3,),
        include_top =False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape = IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training = False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"]
    )
    return model


if __name__ == "__main__":
    train_ds, val_ds, class_names = load_data()

    model = build_model(num_classes=len(class_names))

    model.summary()

    os.makedirs("models", exist_ok=True)

    checkpoint_path = "models/tf_ripeness_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    EPOCHS = 5
    print("\nStarting Model Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    print(f"\nTraining complete! Best model saved to {checkpoint_path}")