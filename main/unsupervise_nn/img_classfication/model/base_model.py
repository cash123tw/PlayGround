import tensorflow as tf
import tensorflow.keras.layers as layers

from main.unsupervise_nn.img_classfication.model import distance_layer

class SiameseModel(tf.keras.Model):
    def __init__(self,siamese_net,margin=0.5):
        super(SiameseModel,self).__init__()

        self.net_work = siamese_net
        self.margin = margin
        self.loss_tracker = tf.metrics.Mean(name='custom_loss')

    def call(self,inputs,**kwargs):
        return self.net_work(inputs)

    #每個 train step 後會呼叫
    def train_step(self,data):
        #紀錄 gradient
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        #計算梯度
        gradients = tape.gradient(loss,self.net_work.trainable_weights)

        #梯度下降
        optimizer:tf.optimizers.Optimizer = self.optimizer
        optimizer.apply_gradients(
            zip(gradients,self.net_work.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self,data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    #triplet loss func
    #L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    def _compute_loss(self,data):
        (ap,an) = self.net_work(data)

        loss = ap - an
        loss = tf.maximum(loss + self.margin,0.)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

class VGG16EmbeddingModel():
    def embedding(self):
        base_cnn = tf.keras.applications.resnet.ResNet50(
            weights="imagenet", input_shape=(None,None) + (3,), include_top=False
        )

        vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,input_shape=(None, None, 3))
        vgg16.trainable = False

        conv2d = layers.Conv2D(filters=16, kernel_size=1, strides=1, activation='relu', padding='same')
        global_max = layers.GlobalMaxPooling2D()
        flat = layers.Flatten()

        output = conv2d(vgg16.output)
        output = global_max(output)
        output = flat(output)

        self.base_model = vgg16
        self.inp = vgg16.inputs
        self.out = output
        self.shared_model = tf.keras.Model(inputs=vgg16.inputs,outputs=[output],name='embedding')

        return self.shared_model

    def vgg16_trainable(self,flag=False):
        self.base_model.trainable = flag
        print('setting vgg16 to trainable %s'%flag)

