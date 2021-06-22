
import tensorflow as tf

def pixel_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return tf.reduce_mean(tf.reduce_mean(tf.abs(y_true - y_pred), axis = 0))


def RalitivisticAverageLoss(non_trans_disc, type_ = "G"):
    loss = None

    def D_Ra(x, y):
        return non_trans_disc(x) - tf.reduce_mean(non_trans_disc(y))

    def loss_D(y_true, y_pred):
        real_logits = D_Ra(y_true, y_pred)
        fake_logits = D_Ra(y_pred, y_true)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_logits), logits=real_logits))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_logits), logits=fake_logits))
        
        return real_loss + fake_loss
    
    def loss_G(y_true, y_pred):
        real_logits = D_Ra(y_true, y_pred)
        fake_logits = D_Ra(y_pred, y_true)
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(real_logits), logits=real_logits)
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_logits), logits=fake_logits)
        
        return real_loss + fake_loss
    

    return loss_G if type == "G" else loss_D


def PerceptualLoss(weight = None, input_shape = None, loss_type = "L1"):
    vgg_model = tf.keras.applications.vgg19.VGG19(
        input_shape = input_shape, weights = weight, include_top = False
    )

    for layer in vgg_model.layers:
        layers.trainable = False
    vgg_model.get_layer("block5_conv4").activation = lambda x: x
    phi = tf.keras.Model(
        inputs = [vgg_model.input],
        outputs =[vgg_model.get_layer("block5_conv4").output])
