import os 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np




DATA_DIR = "data/chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = (224, 224)   # MobileNetV2's expected input size
BATCH_SIZE = 32
EPOCHS = 10             # kept modest given the time constraint
VALIDATION_SPLIT = 0.2 


def print_data_overview():
    print("=" * 50)
    print("DATA OVERVIEW")
    print("=" * 50)
    for cls in ["NORMAL", "PNEUMONIA"]:
        train_count = len(os.listdir(os.path.join(TRAIN_DIR, cls)))
        test_count = len(os.listdir(os.path.join(TEST_DIR, cls)))
        print(f"  {cls}: {train_count} train images, {test_count} test images")
    print() 

def build_data_generators():
    """
    ImageDataGenerator handles preprocessing AUTOMATICALLY as images are
    loaded from disk during training — we don't manually edit any files.
 
    For TRAINING data, we apply:
      - rescale=1./255        -> normalize pixel values from [0,255] to [0,1]
      - rotation/zoom/flip     -> data augmentation (helps with class imbalance
                                   and helps the model generalize better)
      - validation_split=0.2   -> automatically carves 20% of train/ out for
                                   validation (since the provided val/ folder
                                   only has 16 images, too small to use)
 
    For TEST data, we ONLY rescale — no augmentation. We want to evaluate
    on real, unmodified images, not artificially altered ones.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT,
    )
 
    test_datagen = ImageDataGenerator(rescale=1./255)
 
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )
 
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )
 
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
 
    print(f"Class label mapping: {train_generator.class_indices}")
    # e.g. {'NORMAL': 0, 'PNEUMONIA': 1} — confirms which label is which
 
    return train_generator, val_generator, test_generator


def build_model():
    """
    Load MobileNetV2 pretrained on ImageNet (1.4 million general photos),
    but REMOVE its original final classification layer (include_top=False)
    since that layer was built for 1000 unrelated classes (cats, cars, etc).
 
    We freeze the pretrained layers (they already know how to detect
    edges, textures, shapes) and add our OWN small classification head
    on top, trained specifically to distinguish NORMAL vs PNEUMONIA.
    """
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze pretrained layers
 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)  # reduces overfitting, remember our earlier discussion
    output = Dense(1, activation="sigmoid")(x)  # binary output: 0 or 1
 
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def train_model(model, train_gen, val_gen):
    """
    Handle class imbalance with class_weight — this tells the model to
    "pay more attention" to mistakes on the minority class (NORMAL),
    since PNEUMONIA outnumbers it ~3:1 in training data.
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights (to counter imbalance): {class_weight_dict}")
 
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
    ]
 
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )
    return history


def evaluate_model(model, test_gen):
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 50)
 
    predictions = model.predict(test_gen)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    true_labels = test_gen.classes
 
    print("\nClassification report (precision/recall/F1 per class):")
    print(classification_report(true_labels, predicted_labels,
                                  target_names=["NORMAL", "PNEUMONIA"]))
 
    print("Confusion matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    print("(rows = actual, columns = predicted; order: NORMAL, PNEUMONIA)")
 

if __name__ == "__main__":
 print_data_overview()
 train_gen, val_gen, test_gen = build_data_generators()
 model = build_model()
 print(model.summary())
 history = train_model(model, train_gen, val_gen)
 evaluate_model(model, test_gen)
 model.save("pneumonia_model.h5")
 print("\nModel saved to pneumonia_model.h5")    