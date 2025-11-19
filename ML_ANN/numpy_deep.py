import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from PIL import Image
    import tensorflow as tf
    import matplotlib.pyplot as plt
    return mo, np, plt, tf


@app.cell
def _(np):
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)
    return (a,)


@app.cell
def _(a, mo):

    mo.md(f'''a.shape : prints the array shape — a tuple (rows, columns). For this array it prints **{a.shape}**

    a.ndim : number of dimensions (axes). Here it prints **{a.ndim}** (2D array).

    a.size : total number of elements in the array. Here it prints **{a.size}** (6 elements).

    a.dtype : data type of the elements in the array. Here it prints **{a.dtype}** (integers of 64 bits).

    a.itemsize : size in bytes of each element in the array. Here it prints **{a.itemsize}** (8 bytes per element).

    a.nbytes : total size in bytes of the array. Here it prints **{a.nbytes}** (48 bytes).

    ''')
    return


@app.cell
def _(np):
    d3 = np.array([[[1, 2, 3], [4, 5, 6]],
                   [[7, 8, 9], [10, 11, 12]],
                   [[13, 14, 15], [16, 17, 18]]])
    print(d3.shape)
    # (3, 2, 3) : 3 layers, each with 2 rows and 3 columns
    return (d3,)


@app.cell
def _(np):
    d3a = np.array([[[1], [4]],
                   [[7], [10]],
                   [[13], [16]]])
    print(d3a.shape)
    # (3, 2, 1) : 3 layers, each with 2 rows and 1 column

    return


@app.cell
def _(d3):
    # chain indexing to access elements in a 3D array
    print(d3)
    print(d3[2][0][1])  # Output: 14
    # Accessing the element '14' located in the 3rd layer, 1st row, 2nd column

    # Multi-dimensional indexing
    print(d3[2, 0, 1])  # Output: 14
    return


@app.cell
def _(np):
    d2 = np.array([[1, 2, 3,4], 
                   [5, 6, 7,8], 
                   [9,10,11,12], 
                   [13,14,15,16]])
    # array[start:end:step]
    print(d2[2:4]) # print(d2[:4])
    # prints [[ 9 10 11 12]
    #         [13 14 15 16]]    
    # slicing rows from index 2 to 4 (excluding 4) 
    print(d2[:2])

    print(d2[0:4:2])
    #prints [[ 1  2  3  4]
    #        [ 9 10 11 12]]
    # slicing rows from index 0 to 4 (excluding 4) with a step of 2 
    print(d2[::2])
    print(d2[::-1])  # reverses the array rows-wise
    print(d2[::-2]) # reverses the array rows-wise with a step of 2
    return (d2,)


@app.cell
def _(d2):
    print(d2[:, 2]) # prints [ 3  7 11 15] prints all rows in the 3rd column
    print(d2[:, 1:3]) 
    print(d2[:, 1::2])  # prints all rows from index 1 to end with a step of 2
    return


@app.cell
def _(np):
    empty_array = np.empty(5, np.int32)
    print(empty_array)  # Output may contain arbitrary values
    return


@app.cell
def _(np):
    zero_arr = np.zeros(5 , np.int32)
    print(zero_arr) # Output: [0 0 0 0 0]
    return


@app.cell
def _(np):
    zero_array = np.zeros((3,5) , np.int32)
    print(zero_array)
    return (zero_array,)


@app.cell
def _(np, zero_array):
    ones_array = np.ones((5,4) , np.int32)
    print(ones_array)
    another_ones = np.ones_like(zero_array)  # creates an array of ones with the same shape as zero_array
    print(another_ones)
    return


@app.cell
def _(np, plt):
    linear = np.linspace(1, 10, 15)  # 15 values from 0 to 10
    print(linear)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(linear, marker='X')
    fig
    return


@app.cell
def _(np):
    r = np.random.randint(1, 100, size=5) # 5 random integers between 1 and 100
    print(r)
    return


@app.cell
def _(np):
    r1 = np.random.randint(1, 100, size=(3,5)) # 3x5 array of random integers between 1 and 100
    print(r1)   
    r2 = r1.reshape(5,3)  # reshaping the array to 5x3
    print(r2)
    r3 = r1.reshape(15)
    print(r3)  # flattening the array to 1D with 15 elements
    return


@app.cell
def _(np):
    txdata = np.loadtxt('tempdata/img_output.txt', dtype=np.int32)
    print(txdata)
    print(txdata.shape)
    print(txdata.size)

    return (txdata,)


@app.cell
def _(plt, txdata):
    # fig1, ax1 = plt.subplots(figsize=(10, 6))
    # ax1.imshow(txdata)
    # ax1.imshow(txdata, cmap='gray')

    # fig1
    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0,0].imshow(txdata)
    axes[0,0].set_title("Image 1")

    axes[0,1].imshow(txdata, cmap='gray')
    axes[0,1].set_title("Image 2")

    # axes[1,0].imshow(img3)
    # axes[1,0].set_title("Image 3")

    # axes[1,1].imshow(img4)
    # axes[1,1].set_title("Image 4")

    for ax1 in axes.ravel():
        ax1.axis("off")
    fig1
    return (axes,)


@app.cell
def _(np):
    npydata = np.load('tempdata/img_output.npy')
    print(npydata.size)
    print(npydata.shape)
    return (npydata,)


@app.cell
def _(axes, npydata):
    # fig3, ax3 = plt.subplots(figsize=(10, 6))
    # ax3.imshow(npydata)

    # fig3

    axes[1,0].imshow(npydata)
    axes[1,0].set_title("Image 3")
    return


@app.cell
def _(mo, np, plt, tf):
    def linear_regression():

        # Generate some linear data with noise
        trX = np.linspace(-1, 1, 101) 
        trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

        # Define the linear model
        def model(X, w):
            return tf.multiply(X, w) # lr is just X*w so this model line is pretty simple

        #
        w = tf.Variable(0.0, name="weights") # create a shared variable for the weight matrix
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        # Training loop
        for i in range(100):
            for (x, y) in zip(trX, trY):
                with tf.GradientTape() as tape:
                    y_model = model(x, w)
                    cost = tf.square(y - y_model) # use square error for cost function

                gradients = tape.gradient(cost, [w])
                optimizer.apply_gradients(zip(gradients, [w]))

        print(w.numpy())  # It should be something around 2

        # Plot the training data and learned line
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(trX, trY, alpha=0.6, label='Training data (noisy)')
        ax.plot(trX, 2 * trX, 'r--', label='True line (y = 2x)', linewidth=2)
        ax.plot(trX, w.numpy() * trX, 'g-', label=f'Learned line (y = {w.numpy():.4f}x)', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Linear Regression: Learning y = 2x from Noisy Data')
        ax.legend()
        ax.grid(True, alpha=0.3)

        mo.md(f"**Learned weight: {w.numpy():.4f}** (target: 2.0)")
        fig

    linear_regression()
    return


@app.cell
def _(np, tf):
    def logistic_regression():
        # Load MNIST from keras datasets (TF2-compatible)
        (trX_int, trY_int), (teX_int, teY_int) = tf.keras.datasets.mnist.load_data()

        # Preprocess: flatten and normalize
        trX = trX_int.reshape(-1, 784).astype(np.float32) / 255.0
        teX = teX_int.reshape(-1, 784).astype(np.float32) / 255.0

        # Convert labels to integer arrays and one-hot for loss
        trY_int = trY_int.astype(np.int32)
        teY_int = teY_int.astype(np.int32)
        trY_oh = tf.one_hot(trY_int, depth=10, dtype=tf.float32)

        # Weight initializer compatible with TF2
        def init_weights(shape):
            return tf.Variable(tf.random.normal(shape, stddev=0.01, dtype=tf.float32), name="w")

        # Linear model producing logits
        def model(X, w):
            return tf.matmul(X, w)

        # Initialize weights and optimizer
        w = init_weights([784, 10])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

        batch_size = 128
        epochs = 100
        num_samples = trX.shape[0]

        # Training loop (minibatch SGD using GradientTape)
        for epoch in range(epochs):
            # Shuffle dataset each epoch
            idx = np.random.permutation(num_samples)
            trX_shuffled = trX[idx]
            trY_shuffled = trY_oh.numpy()[idx]

            for start in range(0, num_samples, batch_size):
                x_batch = trX_shuffled[start:start+batch_size]
                y_batch = trY_shuffled[start:start+batch_size]

                x_batch_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_batch_tf = tf.convert_to_tensor(y_batch, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    logits = model(x_batch_tf, w)  # shape (batch, 10)
                    per_example_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_tf, logits=logits)
                    loss = tf.reduce_mean(per_example_loss)

                grads = tape.gradient(loss, [w])
                optimizer.apply_gradients(zip(grads, [w]))

            # Evaluate on test set after epoch
            te_logits = model(tf.convert_to_tensor(teX, dtype=tf.float32), w)
            preds = tf.argmax(te_logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, teY_int), tf.float32))

            print(f"Epoch {epoch:03d}: test accuracy = {accuracy.numpy():.4f}")

    # Run logistic regression
    logistic_regression()
    return


if __name__ == "__main__":
    app.run()
