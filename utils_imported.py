import tensorflow.keras.backend as K

def get_wbfc(alpha=.8, gamma=2, mulcoef=1):
    def weighted_binary_focal_crossentropy(target, output):
        """
        target, output - arrays
        alpha - float in (0,1); weight of class '1'
        gamma - focusing parameter
        mulfactor - multiply coefficient
        """
        loss = -alpha * target * K.log(output) * (1-output)**gamma - \
            (1-alpha) * (1-target) * K.log(1-output) * output**gamma
        return mulcoef * K.mean(loss)
    
    return weighted_binary_focal_crossentropy
