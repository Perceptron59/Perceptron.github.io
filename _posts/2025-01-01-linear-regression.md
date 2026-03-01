---
title: "The Essence of Linear Regression"
date: 2025-01-01
layout: single
---

The Essence of Linear Regression

	Machine Learning 
Before we even dive into the world of Machine Learning models, we must understand what Machine Learning even is. The official definition says “it is the field of study that gives computers the ability to learn from data without being explicitly programmed”. But there is another way to look at this definition.
One of the most fundamental properties of our entire universe including humans is patterns, and one of the most fundamental properties of the human brain is pattern recognition. We see and build patterns every day, all the way from arrangement of quarks in subatomic particles to galaxies, in architecture and arrangement of benches or tiles, we observe and use these patterns every day, in fact our brains make them up even when they do not exist. The way we recognise these patterns is through our senses of hearing, seeing, feeling, smelling, etc. This is what machine learning is trying to do, to recognise these patterns in our everyday world and gain insights from them and help humans make better decisions, understand concepts better, solve problems efficiently, most importantly make predictions about the future and many other things. Of course, these algorithms cannot recognise these patterns in a human way so they do so in the next best way, through mathematics. That is what machine learning is, finding these patterns in some data through mathematics and algorithms.
And today we will take a look at one of the most basic and famous machine learning algorithms which is linear regression. I will be trying my best to dilute the math into understandable words for someone who does not understand the math behind linear regression.
	ML models in words
Instead of directly going into the technicalities, we can first try to understand what a machine learning model looks like and make a gradual entry into the concepts.
Learning = Representation + Evaluation + Optimization
This simple equation captures the core structure of most machine learning models, we can break it down and understand each term in the equation:
	Representation: Any problem that you want to solve will always have a predefined structure, and after analysing the dataset you have to make the choice about the model that will best recognise the patterns in the given dataset, every model is suitable for solving a specific kind of problem in a dataset, some will predict, some will classify, some will cluster similar data points, some will look at the most complex relationships and give us insights. It all comes down to what model you think will represent your data best.
	Evaluation: After analysing the dataset and choosing some model you will train the model. You will have a training dataset – a dataset with both, questions and answers - which you will use to train the model. When the model makes some prediction, we need to know how accurate the model is. This is done by comparing predicted values with the true values we have from the training data, technically it is called the loss function of the model. Our objective is to make sure that the predictions of the model are as close as possible to the real answers we have, to make this loss as small as possible. For this we need optimization.
	Optimization: Optimization is the process of gradually bettering the model step by step (iteratively). There are numerous methods which tweak the inner workings of the model - namely the parameters of the model - which try to shift the predictions of the model close to the actual answers.

We will see how these three steps show up naturally in linear regression.

	Setting up linear regression
We cannot dive into the mathematics before getting some notations and assumptions out of the way. We can do this by taking a classic example of house prices.
Area (x_1) (sq. feet)	No. of bedrooms (x_2)	Price (y in 1000 $)
2104	3	400
1600	3	330
2400	3	369
1416	2	232
3000
⋮
⋮
⋮	4
⋮
⋮
⋮	540
⋮
⋮
⋮


	Notations
	x_1 and x_2 here are known as the features or input variables. We can have n features in a dataset. In the real world these features can be related to each other, for example the number of bedrooms will increase as the area of the house increases. Linear regression does not necessarily need independence between features but highly correlated features can make interpretation of the learned parameters more difficult (we will see what parameters are ahead).
	y is the output or the target variable, this is the variable which we have to predict. 
	Here we see just 5 examples but we can have m training examples in our dataset.
	A single training example can be written as (x^((i) )ⓜ,y^((i) ) ), where x^((i) )is the feature vector of the i-th example and y^((i) )is its corresponding output.
Each feature inside this vector is written as x_j^((i) ), where j = 1, 2, …, n and i = 1, 2, …, m.
Thus, the dataset consists of m such training examples. Keep in mind the superscript is an indicator of the example number and not a power for x or y, the subscript is an indicator for the feature number.
	Linear Regression
Now we come to the core of this post, we derive the linear regression model, and any good derivation starts with some assumptions about the world, then we declare some variables and then we see where the math takes us.
Regression is all about mapping the features to their outputs, trying to understand how each feature is affecting the output. In our example that would mean how the area of the house or the number of bedrooms affects the price of the house. The assumption here is that the output is a linear combination of input features, meaning that we are assuming the relationship is linear in the parameters. We could of course take any other relationship here but then we would move away from linear regression and step into the worlds of other models. Mathematically this linear relationship looks like:
h_θ (x)= θ_0+ θ_1 x_1+ θ_2 x_2
 
Geometrically, this represents a line (in one dimension), a plane (in two dimensions), or more generally a hyperplane in higher dimensions. Another bit of notation here would be θ_j which is called the weight or the parameters of the model. They tell us how strongly each feature influences the outcome (hence the word “Weight” for each feature). In the real world though, we turn to matrices to represent this equation in a similar way. That would look like:

θ=[█(■(θ_0@θ_1@θ_2@⋮)@θ_j )] and  x=[█(■(1@x_1@x_2@⋮)@x_j )]
Notice how x_0=1, so our θ_0 fits naturally into our dot product.
Our hypothesis now becomes:
h_θ (x)= θ^T x

This compact form makes the mathematics cleaner and aligns naturally with how we implement the model in code.
Now if you look back you will notice we have done our representation part. We can now move on to Evaluation. 

	Evaluation (Cost Function)
Now this is where things need to be properly understood. 
We have chosen how we want to model the relationship between the features and the outcome. We also declared some parameters to help us understand the relationship between the feature and the outcome. The question remains – How do we choose our parameters? – that we need to tackle now.
To choose the best parameters we need to know the predicting status of the model first. Namely the error or the cost function. The cost function tells us how far our model’s predictions are from the true outputs. Let us say I just declared some random numbers for all the values of θ, we will get some absurd prediction from the model. Then we compare that prediction (or hypothesis) to the real answer with the formula as follows:

error= h_θ (x^((i) ) )- y^((i) )

This error is just for one specific example, we would of course need to aggregate the error over all the m examples in our dataset, which looks like this:
J(θ)= 〖1/2m ∑_(i=1)^m▒( h_θ (x^((i) ) )- y^((i) ))〗^2
or
J(θ)= 〖1/2m ∑_(i=1)^m▒( θ^T x^((i))- y^((i) ))〗^2



We can break this equation down:
	Multiplying the cost by a positive constant does not change the location of the minimum. The  1/2  is included so that when we differentiate later, the 2 from the square cancels out.
	1/m keeps our scaling consistent. Helps us with interpretation, this is why we called it the Mean Squared Error (MSE). This ensures that the error does not increase just because we are increasing the amount of data
	We square our error from before to avoid the cancellation of our negative and positive errors.
Let us say we get two errors, +10 and -10, unless we square them, they will nullify each other and we will percept it as zero error. 
	Squaring penalizes larger error more. For example, if error is 5 then 5^2 would be 25, that forces the model to focus on larger errors more explicitly because it is punished more for larger mistakes. It makes the model sensitive to outliers.
	Squaring helps us smoothen the cost function which in turn helps us with taking its derivatives in the future. Absolute error is not differentiable at 0.
	Geometrically, squaring the error makes the term quadratic, which is a parabola in 1D, a bowl in 2D and a convex quadratic surface in n dimensions. Convexity is important here as it guarantees a unique global minimum on the surface which would be the lowest error or the best values of θ. We will look deeper into this point in a bit

	Vector Form
This same cost function can be written in vector form as follows:
J(θ)= 〖1/2m  ||Xθ-y||〗^2
	Predicted Vector: Xθ
	Outcome Vector: y
We calculate the squared Euclidean Distance between these two vectors to find the error vector.
So linear regression becomes: find the vector Xθ that is closest to y in Euclidean space.
	Optimization
We have now got the framework of how we can check if the model is a good predictor or not.
Now we need to make sure the model is indeed a good predictor, we do so by going through a process of selecting the best parameters that will reduce the error. These methods are namely:
	Gradient Descent
	Normal Equation
There are of course other methods but we can focus on these ones for now.
	Gradient Descent
To start with this, I would first need to give you a mathematical and geometrical framework, so that we can see this process happening. 
Recall that I told you about how squaring the error made it quadratic and convex, a smooth bowl. Geometrically that looks like:
  

Let me help you see what is happening in these figures. 
Our hypothesis is linear in parameters, when we square the error, it becomes quadratic in parameters, producing the bowl surface. Mind you, this geometry is just for two features (consider the housing prices example above) because we cant visualise in higher dimensions.
The X and Y axes are the two parameters of the features θ_1,θ_2. The Z axis is the cost function J(θ). The aim here is to go as low as possible on the Z axis to find the minima which will consequently give us the best values of θ_1 and θ_2. A quick note that all the possible values of θ_1 and θ_2 (X and Y axes), 0 to ∞, is called the hypothesis space.

We now have the visual idea and our aim – to get to the minima of the bowl. How do we get to that minima?
To understand that, we can start in one dimension.
 
We start from a random point on the curve and aim to go down, we happen to be on the left side so our slope which would be a tangent to the point would be negative and moving right would reduce the cost, vice versa if we were on the right side.
This means that slope tells us the direction of steepest increase, negative slope would tell us the direction of steepest decrease.
This is in one dimension, if we move on to higher dimensions, this slope becomes a vector, which is called a gradient.
Same concept is applied here. A gradient points in the direction of the steepest increase, the negative of the gradient would point in the steepest decrease.
With that intuition in mind, let us see this mathematically:
θ □(∶= θ- α∇J(θ))  
This is the update rule for gradient descent. Let us break it down:
	θ in this formula is being updated iteratively
	□("∶=) " here means assignment and not equals (like in python “=” is assignment and “==” means equals)


	θ here is a vector θ=[█(■(θ_0@θ_1@θ_2@⋮)@θ_j )], ∇J(θ) would be the gradient vector, meaning the slope of each particular dimension in the vector, that would look like this:
θ_0:=θ_0-α ∂J/(∂θ_0 )
θ_1:=θ_1-α ∂J/(∂θ_1 )
θ_2:=θ_2-α ∂J/(∂θ_2 )
⋮

	The derivative of the cost function with respect to the parameter is:
∂J/(∂θ_J )=  1/m ∑▒〖〖(h〗_θ (x^((i) ) )- y^((i) ))x_j^((i)) 〗
This is our slope for one parameter, we calculate this for every parameter to get the gradient.
	θ- α∇J(θ) means we take a step in the direction of the steepest decrease (notice we are subtracting the gradient, meaning we move away from the steepest increase). We iteratively repeat this until we see no more improvement in the cost function.
	α is called the learning rate, it is a scaling factor for how big each step would be. We cannot set the learning rate too high as we many overshoot the minima, too low and we will take many more iterations to reach to the minima. A common starting point in simple problems is 0.1, but the optimal learning rate depends on the problem and often requires tuning.
	At each step, we use only local slope information to decide the direction of movement.

Linear Regression has a very unique property which it gets from its quadratic cost function. If you see the figures above, you will notice that there is only one global minimum which we need to get to. Unlike many non-linear models (e.g., neural networks), linear regression with squared loss has no local minima. 
 

Gradient descent is an iterative approach to find this global minimum. 
But since we have only 1 global minimum and no local minima, maybe we can just jump to the global minimum in 1 step?
We do so by the normal equation. This is called the closed form solution of linear regression. 

	Normal Equation
Recall the vector form of our cost function:
J(θ)= 〖1/2m  ||Xθ-y||〗^2
We can write this in another way:
J(θ)=  1/2m  (〖Xθ-y)〗^T (Xθ-y)           
An important matrix identity has been applied here:
(|(|v|)|^2= v^T v)
Now we expand the cost function:
J(θ)=  1/2m(θ^T X^T Xθ-2y^T Xθ+ y^T y)

Take the derivative cost function now, but for this we will need to calculus results for matrices. 
∇_θ (θ^T Aθ)=2Aθ
And when A is symmetric, and our X^T X is indeed symmetric:
∇_θ=(b^T θ)=b
When we apply these identities into the derivative of the cost function we get:
∇J(θ)=  1/m(X^T Xθ- X^T y)
Since the cost function is convex and differentiable, the minimum occurs where its gradient is zero:
∇J(θ)=0
(X^T Xθ- X^T y)=0
X^T Xθ= X^T y
We have finally gotten the normal equation, now we just need to solve for θ:
θ= 〖〖(X〗^T X)〗^(-1) (X^T y)
Provided X^T X is invertible.
We now have the closed form solution.
There is a geometric meaning to this too. X^T (Xθ-y)=0 simply tells us that the residual vector Xθ-y (which is the error in vector form) is orthogonal to the column space of X. Meaning that the error is at the minimum. 

7. Why Linear Regression is Elegant
This model is one of the simplest and still one of the most widely used models in supervised machine learning (labelled dataset). If you are able to master this model in mechanism and understanding you will build a very strong foundation on what machine learning fully entails. 
	The model is linear in its parameters, which makes it analytically tractable and mathematically transparent. 
	The squared loss gives us a smooth convex bowl which is easy to work with and guarantees us a global minimum.
	Quadratic structure of the loss allows us to derive the normal equation which will give us the best parameters in just one step.
	The geometry of this model is fairly easy to understand once you get the hang of it, its interpretation as orthogonal projection provides deep intuition about what the model is actually doing.
	Limitations
Linear regression is of course not perfect, it will have some caveats but so does every other model. Which is why you need to know why you are choosing a particular model before actually using it. Some limitations of Linear Regression are:
	Sensitivity to outliers because of squared error: if there even one outlier way away from the normal dataset, a single large outlier can disproportionately influence the fitted line, because squared error heavily penalizes large deviations.
	Multicollinearity: Multicollinearity occurs when features are highly linearly correlated. In extreme cases, perfect dependence makes X^T X singular and non-invertible, like area of the house and number of bedrooms (Ex. x_1= 〖2x〗_2), that can confuse the model as to how to assign the parameters to each feature if they are dependent on each other. Mathematically X^T X becomes singular and non-invertible.
	What if the data is not linear? Our assumption here has been that the data is linear, but of course in the real world the data can follow any curved path. We cannot fit a straight line to a curved dataset. This can often be addressed by expanding the feature space (e.g., polynomial features), though truly complex relationships may require more expressive models. 
	The normal equation works only for smaller datasets with limited amounts of features, otherwise it becomes too computationally expensive. Normal equation requires computing (X^T X)^(-1) ┤, which is roughly O(n^3)in number of features. We have alternatives too like stochastic gradient descent or batch gradient descent which scale better for larger n. Each path has their trade-offs and we must choose carefully.
	Few Clarifications
This post focused on the structural essence of linear regression, truth is that the model can go much deeper than this, but since this is serving as an introduction to this model and machine learning in general, I decided to keep it simple. Topics such as probabilistic interpretations, regularization, and statistical assumptions (e.g., Gaussian noise) can deepen this framework further.
The best suggestion would be to understand the math behind everything, matrix identities, orthogonality, gradients etc. these are little concepts that, if missing, you will see yourself losing clarity. Another suggestion would also be to code all of this in python. The sources are all over the internet and YouTube. Then you can move onto deeper concepts of machine learning in general. 

