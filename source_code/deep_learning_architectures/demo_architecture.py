import tensorflow as tf

class demo_architecture(tf.keras.models.Sequential):
    
    """
    mode = ("binary", "classification", "regression")
    Utiliza Convoluciones 1D, intercaladas con capas max pooling.
    Finaliza con flatten y dense.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()
        
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))

        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64,
            activation="tanh"))
        
        if mode == "binary":
            self.add(tf.keras.layers.Dense(units=1,
                activation="sigmoid"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy())
        if mode == "classification":
            self.add(tf.keras.layers.Dense(units=len(labels),
                activation="softmax"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy())
        if mode == "regression":
            self.add(tf.keras.layers.Dense(units=1,
                activation="linear"))
            self.compile(optimizer=tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.MeanSquaredError())