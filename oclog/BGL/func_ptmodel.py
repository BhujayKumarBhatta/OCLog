import tensorflow as tf
import numpy as np
import tensorflow as tf
tf.random.set_seed(123)

def get_embedding_layer(bglog):    
    tk = bglog.tk
    vocab_size = len(tk.word_index)
    print(f'vocab_size: {vocab_size}')
    char_onehot = vocab_size
    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))
    for char, i in tk.word_index.items(): # from 1 to 51
        onehot = np.zeros(vocab_size)
        onehot[i-1] = 1
        embedding_weights.append(onehot)
    embedding_weights = np.array(embedding_weights)
    return embedding_weights, vocab_size, char_onehot

def pt_func_model(train_data,validation_data, bglog, conv1d_set1 = 3, conv1d_set2 = 3, dense_neurons=2048, filters=64, kernel_size=3,maxpool_1=True,epochs=1,):
    embedding_weights, vocab_size, char_onehot = get_embedding_layer(bglog)
    B = train_data.element_spec[0].shape[0]
    inputs = tf.keras.layers.Input(batch_shape=(B, train_data.element_spec[0].shape[1], train_data.element_spec[0].shape[2]), dtype='float64' )
    x = tf.keras.layers.Embedding(input_dim=vocab_size+1,
                                    output_dim=vocab_size,
                                    input_length=train_data.element_spec[0].shape[2],
                                    weights = [embedding_weights],
                                    )(inputs)
    for _ in range(conv1d_set1):
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    if maxpool_1:
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, train_data.element_spec[0].shape[2]))(x)
        x = tf.reshape(x, (B, train_data.element_spec[0].shape[1], filters))        
        for _ in range(conv1d_set2):
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=(train_data.element_spec[0].shape[1]) )(x)
        x = tf.reshape(x, (B, filters))
    if not maxpool_1:
        x = tf.keras.layers.Flatten()(x)       
    x = tf.keras.layers.Dense(dense_neurons)(x)
    outputs = tf.keras.layers.Dense(train_data.element_spec[1].shape[1], activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    hist = model.fit(train_data, validation_data=validation_data, epochs=epochs) 
    return model, hist

