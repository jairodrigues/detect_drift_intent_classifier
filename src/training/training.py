import pickle
import tensorflow as tf


BATCH_SIZE = 32
NB_BATCHES = 0
NB_BATCHES_TEST = NB_BATCHES // 5
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 5
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
NB_EPOCHS = 10


def Training():

    def __init__(self, model, checkpoint, sorted_all):
        all_data_set = tf.data.Dataset.from_generator(
            lambda: sorted_all, output_types=(tf.int32, tf.int32))
        all_batched = all_data_set.padded_batch(
            BATCH_SIZE, padded_shapes=((3, None), ()), padding_values=(0, 0))
        all_batched.shuffle(len(sorted_all) // BATCH_SIZE)
        test_dataset = all_batched.take(NB_BATCHES_TEST)
        train_dataset = all_batched.skip(NB_BATCHES_TEST)
        Dcnn = model(nb_filters=NB_FILTERS, FFN_units=FFN_UNITS,
                     nb_classes=NB_CLASSES, dropout_rate=DROPOUT_RATE)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        Dcnn.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        history = Dcnn.fit(train_dataset,
                           validation_data=(test_dataset),
                           epochs=NB_EPOCHS,
                           callbacks=[checkpoint.MyCustomCallback()])
        pickle.dump(Dcnn, open('model_dcnn.sav', 'wb'))
        return history
