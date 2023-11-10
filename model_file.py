from keras import layers, models

class Yugi:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_model_summary(self):
        return self.model.summary()

    def train(self, train_data, train_labels, epochs, batch_size, callbacks=None):
        return self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = models.load_model(filename)