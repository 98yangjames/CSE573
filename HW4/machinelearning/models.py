import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        val = nn.as_scalar(self.run(x))
        if val >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        updateWeight = True
        while updateWeight:
            #assume it's great.
            updateWeight = False
            for x, y in dataset.iterate_once(batch_size=1):
                #if f(x) != y*
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(nn.Constant(nn.as_scalar(y)*x.data), multiplier=1)
                    #since we updated we gotta start over
                    updateWeight = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.first_weight = nn.Parameter(1, 50)
        self.first_b = nn.Parameter(1, 50)
        self.second_weight = nn.Parameter(50, 25)
        self.second_b = nn.Parameter(1, 25)
        self.third_weight = nn.Parameter(25, 1)
        self.third_b = nn.Parameter (1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        first = nn.AddBias(nn.Linear(x, self.first_weight), self.first_b)

        second = nn.AddBias(nn.Linear(nn.ReLU(first), self.second_weight), self.second_b)
        third = nn.AddBias(nn.Linear(nn.ReLU(second), self.third_weight), self.third_b)
        # fourth = nn.AddBias(nn.Linear(nn.ReLU(third), self.fourth_weight), self.fourth_b)
        return third
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning = True
        initialize_arr = []
        while learning:
            for x, y in dataset.iterate_once(10):
                learning = False
                if(nn.as_scalar(self.get_loss(x, y)) <= 0.01):
                    learning = False
                else:
                    learning = True
                    initialize_arr = [self.first_weight, self.first_b, self.second_weight, self.second_b, self.third_weight, self.third_b]
                    gradients = nn.gradients(self.get_loss(x,y), initialize_arr)
                    self.first_weight.update(gradients[0], -.001)
                    self.first_b.update(gradients[1], -.001)
                    self.second_weight.update(gradients[2], -.001)
                    self.second_b.update(gradients[3], -.001)
                    self.third_weight.update(gradients[4], -.001)
                    self.third_b.update(gradients[5], -.001)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.first_weight = nn.Parameter(784, 300)
        self.first_b = nn.Parameter(1, 300)
        self.second_weight = nn.Parameter(300, 150)
        self.second_b = nn.Parameter(1, 150)
        self.third_weight = nn.Parameter(150, 75)
        self.third_b = nn.Parameter(1, 75)
        self.fourth_weight = nn.Parameter(75, 10)
        self.fourth_b = nn.Parameter(1, 10)
        self.parameters = [self.first_weight, self.first_b, self.second_weight, self.second_b, self.third_weight, self.third_b, self.fourth_weight, self.fourth_b]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.first_weight),
                                self.first_b))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.second_weight), self.second_b))
        third_layer = nn.ReLU(nn.AddBias(nn.Linear(second_layer, self.third_weight), self.third_b))
        fourth_layer = nn.AddBias(nn.Linear(third_layer, self.fourth_weight), self.fourth_b)
        return fourth_layer
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning = True
        while learning:
            for x, y in dataset.iterate_once(100):
                gradients = nn.gradients(self.get_loss(x, y), self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(gradients[i], -.1)
            if(dataset.get_validation_accuracy() <= 0.98):
                learning = True
            else:
                learning = False
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(47, 300)
        self.b1 = nn.Parameter(1, 300)
        self.xw = nn.Parameter(47, 300)
        self.hw = nn.Parameter(300, 300)
        self.b2 = nn.Parameter(1, 300)
        self.outputw = nn.Parameter(300, 5)
        self.outputb = nn.Parameter(1, 5)
        self.parameters = [self.w1, self.b1, self.xw, self.hw, self.b2, self.outputw, self.outputb]


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #feed forward network:


        first = True
        for char in xs:
            if(first == True):
                h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w1), self.b1))
                first = False
            else:
                char_weight = nn.Linear(char, self.xw)
                h_weight = nn.Linear(h, self.hw)
                add_neural = nn.Add(char_weight, h_weight)
                bias = nn.AddBias(add_neural, self.b1)
                h = nn.ReLU(bias)

        output = nn.AddBias(nn.Linear(h, self.outputw), self.outputb)
        return output
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning = True
        while learning:
            for x, y in dataset.iterate_once(100):
                gradients = nn.gradients(self.get_loss(x, y), self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(gradients[i], -.1)
            if (dataset.get_validation_accuracy() <= 0.85):
                learning = True
            else:
                learning = False