import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # Would be needed with data
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from tensorflow.keras.callbacks import EarlyStopping # Would be used in tuner.search()
import keras_tuner as kt
import numpy as np
import os

# --- Configuration (Usually data-dependent, but set for structure) ---
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 3 # Surprised, Sad, Neutral
# BATCH_SIZE = 32 # Would be set when data generators are defined

# --- Placeholder for Data Loading and Preprocessing ---
# In a real scenario, you would have your ImageDataGenerators or tf.data.Dataset pipeline here
# For example:
# train_datagen = ImageDataGenerator(...)
# validation_datagen = ImageDataGenerator(...)
# train_generator = train_datagen.flow_from_directory(...)
# validation_generator = validation_datagen.flow_from_directory(...)

# For this script to run without erroring out immediately on data,
# we'll define dummy generators. THESE ARE NOT FOR ACTUAL TRAINING.
def dummy_data_generator(num_samples, batch_size, img_height, img_width, num_classes):
    """Generates dummy data. NOT FOR ACTUAL TRAINING."""
    while True:
        dummy_images = np.random.rand(batch_size, img_height, img_width, 3).astype(np.float32)
        dummy_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, num_classes, batch_size), num_classes=num_classes
        )
        yield dummy_images, dummy_labels

DUMMY_SAMPLES = 100 # Small number for placeholder
BATCH_SIZE_DUMMY = 4 # Small batch for placeholder

print("NOTE: Using DUMMY data generators. This script demonstrates KerasTuner setup structure ONLY.")
print("Replace dummy_train_generator and dummy_validation_generator with your actual data pipelines.")

dummy_train_generator = dummy_data_generator(DUMMY_SAMPLES, BATCH_SIZE_DUMMY, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)
dummy_validation_generator = dummy_data_generator(DUMMY_SAMPLES // 4, BATCH_SIZE_DUMMY, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)
dummy_steps_per_epoch = DUMMY_SAMPLES // BATCH_SIZE_DUMMY
dummy_validation_steps = (DUMMY_SAMPLES // 4) // BATCH_SIZE_DUMMY


# --- HyperModel Definition ---
class ResNet50HyperModel(kt.HyperModel):
    def build(self, hp):
        # Load ResNet-50 pre-trained on ImageNet, without the top classification layer
        base_model = ResNet50(
            weights='imagenet', # Using ImageNet weights as a starting point
            include_top=False,
            input_tensor=Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        )
        # Base model is initially frozen for head training
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D(name="avg_pool")(x)

        hp_units = hp.Int('units', min_value=256, max_value=1024, step=128, default=512)
        x = Dense(units=hp_units, activation='relu', name="dense_units")(x)
        x = BatchNormalization(name="batch_norm_dense")(x)

        hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.1, default=0.4)
        x = Dropout(hp_dropout, name="dropout_dense")(x)

        predictions = Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile for the initial head training stage (this compilation will be overridden for fine-tuning stage)
        hp_lr_initial = hp.Float('lr_initial', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
        hp_optimizer_initial_choice = hp.Choice('optimizer_initial', values=['adam', 'sgd', 'rmsprop'], default='adam')

        if hp_optimizer_initial_choice == 'adam':
            optimizer_initial = Adam(learning_rate=hp_lr_initial)
        elif hp_optimizer_initial_choice == 'sgd':
            optimizer_initial = SGD(learning_rate=hp_lr_initial, momentum=0.9)
        else: # rmsprop
            optimizer_initial = RMSprop(learning_rate=hp_lr_initial)

        model.compile(
            optimizer=optimizer_initial,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        # This fit function handles the two-stage training for each trial.
        # Args will be (dummy_train_generator)
        # Kwargs will be {'epochs': ..., 'validation_data': dummy_validation_generator, ...}

        # --- STAGE 1: Train the head ---
        print(f"\n--- HyperTrial: Stage 1 (Training Head) ---")
        print(f"HPs: Units={hp.get('units')}, Dropout={hp.get('dropout')}, Optimizer_Initial={hp.get('optimizer_initial')}, LR_Initial={hp.get('lr_initial')}")

        # `epochs` in kwargs is managed by the tuner (e.g., for Hyperband, it varies per bracket)
        # We need to decide how many of these epochs go to head training vs. fine-tuning.
        epochs_for_trial = kwargs.get('epochs', 10) # Total epochs for this specific trial configuration
        epochs_for_head_training = hp.Int('epochs_head', min_value=3, max_value=10, default=5)
        epochs_for_head_training = min(epochs_for_head_training, epochs_for_trial -1) # Ensure at least 1 for fine-tuning if possible
        if epochs_for_head_training < 1 : epochs_for_head_training = 1


        # Get callbacks from kwargs for the fit method
        trial_callbacks = kwargs.pop('callbacks', []) # Remove callbacks from kwargs to pass them explicitly

        history_initial = model.fit(
            *args, # e.g., dummy_train_generator
            epochs=epochs_for_head_training,
            validation_data=kwargs.get('validation_data'), # e.g., dummy_validation_generator
            steps_per_epoch=kwargs.get('steps_per_epoch'),
            validation_steps=kwargs.get('validation_steps'),
            callbacks=trial_callbacks,
            verbose=kwargs.get('verbose', 2) # Less verbose for tuner search
        )
        if not history_initial.history['val_accuracy']: # Check if training even ran
             return history_initial # Or some indicator of failure

        # --- STAGE 2: Fine-tuning ---
        print(f"\n--- HyperTrial: Stage 2 (Fine-tuning Base Model) ---")
        base_model = model.get_layer('resnet50') # Get ResNet50 base by its default name
        base_model.trainable = True

        hp_unfreeze_blocks = hp.Int('unfreeze_blocks', min_value=1, max_value=3, step=1, default=1)
        if hp_unfreeze_blocks == 1: fine_tune_at = len(base_model.layers) - 33 # Approx conv5
        elif hp_unfreeze_blocks == 2: fine_tune_at = len(base_model.layers) - 80 # Approx conv5 + conv4
        else: fine_tune_at = len(base_model.layers) - 120 # Approx conv5 + conv4 + part of conv3
        fine_tune_at = max(0, fine_tune_at)

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer_idx in range(fine_tune_at, len(base_model.layers)):
            # Keep BatchNormalization layers in inference mode when fine-tuning base,
            # as per Keras guides, to prevent large shifts from disrupting learned features.
            if not isinstance(base_model.layers[layer_idx], BatchNormalization):
                base_model.layers[layer_idx].trainable = True

        hp_lr_fine_tune = hp.Float('lr_fine_tune', min_value=1e-6, max_value=1e-4, sampling='log', default=1e-5)
        hp_optimizer_ft_choice = hp.Choice('optimizer_ft', values=['adam', 'sgd', 'rmsprop'], default='adam')

        if hp_optimizer_ft_choice == 'adam':
            optimizer_fine_tune = Adam(learning_rate=hp_lr_fine_tune)
        elif hp_optimizer_ft_choice == 'sgd':
            optimizer_fine_tune = SGD(learning_rate=hp_lr_fine_tune, momentum=0.9)
        else:
            optimizer_fine_tune = RMSprop(learning_rate=hp_lr_fine_tune)

        model.compile(
            optimizer=optimizer_fine_tune,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        epochs_for_fine_tuning = epochs_for_trial - epochs_for_head_training
        if epochs_for_fine_tuning < 1 and epochs_for_trial > epochs_for_head_training:
            epochs_for_fine_tuning = 1 # Ensure at least one epoch if total allows
        elif epochs_for_fine_tuning < 1: # Not enough total epochs for fine-tuning stage
             print("Skipping fine-tuning stage as epochs_for_head_training >= total epochs for trial.")
             # Return the history from head training as the result for this trial
             # KerasTuner will still use its val_accuracy.
             return history_initial


        history_fine_tune = model.fit(
            *args,
            epochs=epochs_for_fine_tuning, # Use remaining epochs
            initial_epoch=0, # Keras fit initial_epoch is relative to this call
            validation_data=kwargs.get('validation_data'),
            steps_per_epoch=kwargs.get('steps_per_epoch'),
            validation_steps=kwargs.get('validation_steps'),
            callbacks=trial_callbacks,
            verbose=kwargs.get('verbose', 2)
        )
        # KerasTuner automatically monitors the metric specified in `objective` (e.g., 'val_accuracy')
        # from the history object returned by the last `fit` call.
        return history_fine_tune


# --- Instantiate Tuner ---
# Using Hyperband for efficient search
tuner = kt.Hyperband(
    ResNet50HyperModel(),
    objective=kt.Objective("val_accuracy", direction="max"), # Maximize validation accuracy
    max_epochs=20,  # Max total epochs for the longest trial in the last Hyperband bracket
                    # This is split by ResNet50HyperModel.fit into head and fine-tuning stages
    factor=3,
    hyperband_iterations=1, # Number of times to iterate over the full Hyperband algorithm.
                            # Increase for more thorough search but longer time.
    directory='keras_tuner_no_data_demo',
    project_name='emotion_resnet50_hptuning',
    overwrite=True
)

tuner.search_space_summary()

print("\n--- Hyperparameter Search Setup Complete ---")
print("To run the search, you would call tuner.search(...) with your actual data generators.")
print("Example (DO NOT RUN WITHOUT REAL DATA):")
print("# search_callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]")
print("# tuner.search(")
print("#     train_generator,")
print("#     epochs=tuner.max_epochs, # Tuner manages epochs per trial based on max_epochs and factor")
print("#     steps_per_epoch= YOUR_STEPS_PER_EPOCH,")
print("#     validation_data=validation_generator,")
print("#     validation_steps=YOUR_VALIDATION_STEPS,")
print("#     callbacks=search_callbacks")
print("# )")
print("# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]")
print("# print(f'Best HPs: {best_hps.values}')")
print("# best_model = tuner.get_best_models(num_models=1)[0]")
print("# best_model.save('best_hyper_tuned_resnet50_emotion_model.keras')")

# --- Demonstration of building a model with default HPs (no actual tuning here) ---
print("\n--- Building one model instance with default HPs (for structural demonstration) ---")
# Get a set of HPs (e.g., default values defined in the HyperModel)
default_hps = kt.HyperParameters()
# Manually set defaults if not set in hp.Int/Float etc. or to see a specific config
default_hps.Choice('optimizer_initial', values=['adam', 'sgd', 'rmsprop'], default='adam') # Need to define it first
default_hps.Float('lr_initial', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
default_hps.Int('units', min_value=256, max_value=1024, step=128, default=512)
default_hps.Float('dropout', min_value=0.2, max_value=0.6, step=0.1, default=0.4)
default_hps.Int('epochs_head', min_value=3, max_value=10, default=5)
default_hps.Int('unfreeze_blocks', min_value=1, max_value=3, step=1, default=1)
default_hps.Float('lr_fine_tune', min_value=1e-6, max_value=1e-4, sampling='log', default=1e-5)
default_hps.Choice('optimizer_ft', values=['adam', 'sgd', 'rmsprop'], default='adam')


hypermodel_instance = ResNet50HyperModel()
model_instance = hypermodel_instance.build(default_hps)
model_instance.summary()
print("\nModel built with default HPs. This model is not trained or tuned.")
print("Replace dummy data generators and uncomment tuner.search() to perform actual hyperparameter tuning.")