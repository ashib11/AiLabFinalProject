from datasets import load_from_disk
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
from keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Conv2DTranspose, BatchNormalization, Input
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dataset = load_from_disk("chest_xray_pneumonia")


def count_classes(split):
    labels = dataset[split]["label"]
    num_pneumonia = sum(1 for label in labels if label == 1)
    num_normal = sum(1 for label in labels if label == 0)
    return num_pneumonia, num_normal



print("Train set:\n========================================")
train_pneumonia, train_normal = count_classes("train")
print(f"PNEUMONIA={train_pneumonia}")
print(f"NORMAL={train_normal}")

print("\nTest set:\n========================================")
test_pneumonia, test_normal = count_classes("test")
print(f"PNEUMONIA={test_pneumonia}")
print(f"NORMAL={test_normal}")

print("\nValidation set:\n========================================")
val_pneumonia, val_normal = count_classes("validation")
print(f"PNEUMONIA={val_pneumonia}")
print(f"NORMAL={val_normal}")


plt.figure(figsize=(20, 10))
pneumonia_samples = [i for i, x in enumerate(dataset["train"]["label"]) if x == 1][:9]
for i, idx in enumerate(pneumonia_samples):
    plt.subplot(3, 3, i + 1)
    img = np.array(dataset["train"][idx]["image"])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Pneumonia Sample {i + 1}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 10))
normal_samples = [i for i, x in enumerate(dataset["train"]["label"]) if x == 0][:9]
for i, idx in enumerate(normal_samples):
    plt.subplot(3, 3, i + 1)
    img = np.array(dataset["train"][idx]["image"])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Normal Sample {i + 1}")
plt.tight_layout()
plt.show()


normal_sample = next(i for i, x in enumerate(dataset["train"]["label"]) if x == 0)
sample_img = np.array(dataset["train"][normal_sample]["image"])

plt.figure(figsize=(8, 6))
plt.imshow(sample_img, cmap='gray')
plt.colorbar()
plt.title('Single Normal Chest X-Ray')
plt.show()

print("\nImage Statistics:")
print(f"Dimensions: {sample_img.shape[0]}h x {sample_img.shape[1]}w")
print(f"Pixel range: {sample_img.min():.2f}-{sample_img.max():.2f}")
print(f"Mean: {sample_img.mean():.2f}, Std: {sample_img.std():.2f}")


plt.figure(figsize=(12, 6))
sns.histplot(sample_img.ravel(),
             bins=50,
             kde=False,
             stat='count',
             label=f"Mean: {np.mean(sample_img):.2f} ± {np.std(sample_img):.2f}")
plt.legend(loc='upper right')
plt.title('Pixel Intensity Distribution\n(Normal Chest X-Ray)', pad=20)
plt.xlabel('Pixel Intensity (0=Black, 255=White)')
plt.ylabel('Pixel Count')
plt.grid(True, alpha=0.3)
plt.axvline(x=np.mean(sample_img), color='r', linestyle='--', alpha=0.7)
plt.text(x=np.mean(sample_img)+5, y=plt.ylim()[1]*0.9,
         s=f"Mean\n({np.mean(sample_img):.1f})",
         color='r')
plt.tight_layout()
plt.show()


num_pneumonia, num_normal = count_classes("train")
weight_for_0 = num_pneumonia / (num_pneumonia + num_normal)
weight_for_1 = num_normal / (num_pneumonia + num_normal)
class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"Weight of class 0: {class_weight[0]}")
print(f"Weight of class 1: {class_weight[1]}")


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])



model = Sequential()

model.add(data_augmentation)

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))  # <- input_shape here


model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.build(input_shape=(None, 128, 128, 3))

model.summary()


def preprocess_batch(batch):
    images = []
    for img in batch["image"]:
        img = img.resize((128, 128))
        img = np.array(img)
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        images.append(img)

    images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.0
    labels = tf.convert_to_tensor(batch["label"], dtype=tf.int32)

    return {"image": images, "label": labels}



batch_size = 16
dataset = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=batch_size,
    remove_columns=["image"]
)

def create_tf_dataset(split, batch_size=batch_size, shuffle=True):
    return dataset[split].to_tf_dataset(
        columns="image",
        label_cols="label",
        shuffle=shuffle,
        batch_size=batch_size
    )

train_ds = create_tf_dataset("train")
val_ds = create_tf_dataset("validation")
test_ds = create_tf_dataset("test")

steps_per_epoch = len(dataset["train"]) // batch_size
validation_steps = len(dataset["validation"]) // batch_size

checkpoint = ModelCheckpoint('best_model.h5',
                            monitor='val_accuracy',
                            save_best_only=True,
                            mode='max',
                            verbose=1)
early_stop = EarlyStopping(monitor='val_loss',
                          patience=3,
                          restore_best_weights=True)

history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    class_weight=class_weight,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop]
)

best_model = tf.keras.models.load_model('best_model.h5')
best_model.save('pneumonia_model.h5')
print("Best Model saved as pneumonia_model.h5")

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


y_true = []
y_pred = []
for batch in test_ds:
    images, labels = batch
    y_true.extend(labels.numpy())
    y_pred.extend(best_model.predict(images).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


def find_optimal_threshold(y_true, y_pred, method='youden'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    if method == 'youden':
        scores = tpr - fpr
    elif method == 'gmean':
        scores = np.sqrt(tpr * (1 - fpr))
    elif method == 'f1':
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        return thresholds[np.argmax(scores)]

    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]


best_threshold = find_optimal_threshold(y_true, y_pred, method='youden')
print(f"\nOptimal Classification Threshold: {best_threshold:.3f}")
np.save('best_threshold.npy', best_threshold)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred > best_threshold)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title(f'Confusion Matrix (Threshold={best_threshold:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


from sklearn.metrics import classification_report

for threshold in np.arange(0.1, 0.9, 0.1):
    print(f"\nThreshold = {threshold:.1f}")
    preds = y_pred > threshold
    print(classification_report(y_true, preds, target_names=['Normal', 'Pneumonia']))



fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

optimal_idx = np.argmax(tpr - fpr)
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
            label=f'Optimal Threshold ({best_threshold:.2f})')


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

