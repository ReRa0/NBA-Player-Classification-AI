import tensorflow as tf

height = 170
shoesize = 260

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 = a * height + b
    return tf.square(260 - 예측값)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(1000):
    with tf.GradientTape() as tape:
        loss = 손실함수()
    gradients = tape.gradient(loss, [a, b])
    optimizer.apply_gradients(zip(gradients, [a, b]))
    print(a.numpy(), b.numpy())