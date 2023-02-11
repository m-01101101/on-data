---
toc: true
layout: post
description: A gentle introduction to Linear Algebra
categories: [maths for machine learning]
title: An introduction to Linear Algebra
---
# An introduction to Linear Algebra

Algebra is the generic term for the maths of equations in which numbers and operations are written as symbols.

> René Descarte's _La Geometrie_ first introduced standard algebraic notation. When the book was being printed, the printer started to run out of letters. When asked if it mattered if _x_, _y_, or _z_ was used to represent the unknowns, Descarte said it did not. As a result, _x_ became fixed in maths - and the wider culture - as the symbol for the unknown quantity. - _Alex Bellos, Numberland_

The beauty of applied mathematics is that is gives us a new layer of abstraction. In the case of linear algebra that is a language to represent data and manipulate space. In many of the examples, we'll use a geometric lens, i.e. using data to describe a 2D space (an arrow). But mathematics allows us to take these learnings and generalise to $n$ number of dimensions and our data can describe not just space but objects.

We'll look at building an intuition around the concepts and doing some basic work in linear algebra, before passing off the difficult stuff to computers. Though, we'll be intentional about what we're asking the computer to do.

Linear algebra has evolved to a more general and abstract form, representing _vector spaces_. Vectors are used to described points in some finite-dimensional space.  An _n_ dimensional space will be described by a vector with _n_ coordinates. In a 2D world this will be an arrow, with _x_, and _y_ coordinates, the vector describes will be an arrow pointing to B.

Linearity is the property of a mathematical relationship (function) that can be graphically represented as a straight line _- Wikipedia_. This statement will make sense when we start to do operations with vectors. But we'll see how adding and ~scaling~ manipulating the size of vectors is done by moving in a straight line.

It's important to note that linear algebra represents much of the trunk of knowledge required for modern data analysis and many machine learning techniques. Often the use of linear algebra will be implicit, you'll forget you're even using it. Think of it as a prerequisite, a string to your bow, which doesn't get the same attention as it's sexy cousins statistics and probability, but is just as essential.

Take the time to refresh your memory or grasp the concepts here.

## Vectors

Vectors are the building blocks of linear algebra.

Think of a vector as an object that moves us about space (physical or data), it could just be a list of attributes of an object.

Depending on your world view, what exactly you deem to be a vector will vary. Vectors can be considered;

- Vectors are arrows pointing in space

    What defines a vector is its length and the direction it's pointing. You can move it anywhere in space, and it's the same vector.

    A vector on the flat plane are two-dimensional, those in broader space at three-dimensional.

- Vectors are a list of numbers
    
    Vectors are usually viewed by computers as an ordered list of numbers which they can perform "operations" on - such as multiplying by _scalars_ (numbers) to form new vectors.

    The numbers in vectors represent data about an object. The number of dimensions is determined by the length of the vector.

At the start of this learning material, we'll focus on applying a geometric lens to vectors. That is, think of them as an arrow describing dimensions of space, according to some co-ordinate system.

> Vectors can be thought of in a variety of different ways - some geometrically, some algebraically, some numerically. In this way, there are a lot of techniques one can use to deal with vectors.

Concretely, vectors are often a useful way to represent data.

### __Vectors obey two rules__

1. Addition $(r + s == s + r)$ -> _vector addition is associative_

Conceptually you add vectors by moving the tail origin of one vector to the tip of the other. Then drawing a new vector from the tail of the first to the tip of where the second now sits.

You're adding the $x$ and $y$ components together.

$r + s = [r_x + s_x, r_y + s_y]$

![]({{ site.baseurl }}/images/linear_algebra/addingvectors.jpg){:width="300px"}
<br>

In the example above $r = \begin{bmatrix} 1 \\ 2\end{bmatrix}$, $s = \begin{bmatrix} 3 \\ -1\end{bmatrix}$ we can arrive at $t$ geometrically, taking 1 step the right, 2 up, then 3 to the right, and one down.

We add the component parts (component wise), as add the $x$ coordinates and the $y$ coordinates to get our new vector.

1. Multiplication by a scalar (i.e. number, because numbers scale vectors, so we use the terms interchangeably)
   <br>_negative numbers means reverse_

![]({{ site.baseurl }}/images/linear_algebra/vector_multiplication.jpg){:width="250px"}
<br>

We multiple each component in the vector by that scalar.

$r = \begin{bmatrix} i \\ j\end{bmatrix}$ $3r = \begin{bmatrix} 3i \\ 3j\end{bmatrix}$

Vectors give us a language of space and the ability to manipulate space. 
They allow us to represent lots of lists of data together.

__Ties to Machine Learning__

One of the tasks of machine learning is to fit a model to data in order to represent the underlying distribution.

A model allows us to predict the data in a distribution.

We can start with a parameter vector $\mathbf{p}$ and convert it to a vector of expected frequencies $\mathbf{g}_\mathbf{p}$ 

We need a way fit a model's parameters to data and quantify how good that fit is.

One way of doing so is to calculate the "residuals", which is the difference between the measured data and the modelled prediction for each histogram bin.

A better fit would have as much overlap as it can, reducing the residuals as much as possible.

![]({{ site.baseurl }}/images/linear_algebra/residuals.png){:width="400px"}


In the example above, we'd improve the model in orange by decreasing or increasing $\mu$ (the height) and keeping $\sigma$ (the width) roughly the same.

The performance of a model can be quantified in a single number. One measure we can use is the Sum of Squared Residuals, $\mathrm{SSR}$. i.e. we square the difference for each value in actual and predicted, and add all those together;

$$ \mathrm{SSR}_\mathbf{p} = \lVert \mathbf{f} - \mathbf{g}_\mathbf{p} \rVert ^2 $$

<br>

> In the below; orange is the observed, purple the predicts and green the overlap.
```python
μ = 160 ; σ = 6
p = [μ, σ]
histogram(p)
```

![]({{ site.baseurl }}/images/linear_algebra/testing_rss.png){:width="400px"}

```python
μ = 179 ; σ = 7
p = [μ, σ]
histogram(p)
```

![]({{ site.baseurl }}/images/linear_algebra/testing_rss1.png){:width="400px"}

Since each parameter vector $\mathbf{p}$ represents a different bell curve, each with its own value for the sum of squared residuals, $\mathrm{SSR}$, we can draw the surface of $\mathrm{SSR}$ values over the space spanned by $\mathbf{p}$, such as $\mu$ and $\sigma$ in this example.

Every point on this surface represents the SSR of a choice of parameters

![]({{ site.baseurl }}/images/linear_algebra/residuals_as_contours.png){:width="400px"}

We can take a ‘top-down’ view of the surface, and view it as a contour map, where each of the contours (in green here) represent a constant value for the $\mathrm{SSR}$.

![]({{ site.baseurl }}/images/linear_algebra/residuals_as_contours1.png){:width="400px"}

Often we can't see the whole parameter space, so instead of just picking the lowest point, we have to make educated guesses where better points will be.

We can define another vector, $\Delta\mathbf{p}$, in the same space as $\mathbf{p}$ that tells us what change can be made to $\mathbf{p}$ to get a better fit.

For example, a model with parameters $\mathbf{p}'$ = $\mathbf{p}$ + $\Delta\mathbf{p}$ will produce a better fit to data, if we can find a suitable $\Delta\mathbf{p}$.

- Moving at right angles to contour lines in the parameter space is the most effective way to move through the space. Moving along contonour lines has no effect. Moving perpendicular to them can significantly improve or reduce the quality of the fit.

- Moving along the contour lines __does not__ produce the same model

The $\Delta\mathbf{p}$ $\begin{bmatrix}-2 \\ 2\end{bmatrix}$ will give the best improvement in the model below;

![]({{ site.baseurl }}/images/linear_algebra/moving_contours.png){:width="400px"}

$\mu$ and $\sigma$ have to be different, decrease $\mu$ along the x-axis, increase $\sigma$ along the y-axis.

## Simultaneous Equations

Solving simultaneous equations is the process of finding the values of the variables (here $x$ and $y$) that satisfy the system of equations. 

The first goal when solving simple simultaneous equations should be to isolate one of the variables.

For example, using elimination, taking the second equation away from the first to solve the following pair of equations:

$3x−2y=7$

$2x - 2y = 2$

$x = 5, y = 4$

You can use elimination even when the coefficients, the numbers in front of $x$ and $y$, aren't the same.

For example, multiplying both sides of the first equation by 2, then solve using the same method as the last question.

$3x−2y=4$ $\equiv$ $6x - 4y = 8$

$6x + 3y = 15$

$x = 2, y = 1$

A very similar technique can be used to find the inverse of a matrix.

There is also the substitution method, where we rearrange one of the equations to the form $x = ay+b$ or $y = cx+d$ and then substitute $x$ or $y$ into the other equation.

Systems of simultaneous equations can have more than two unknown variables. 
<br>Below there is a system with three; $x$, $y$ and $z$. 

First try to find one of the variables by elimination or substitution, which will lead to two equations and two unknown variables. 

Continue the process to find all of the variables.

$3x − 2y + z =7$

$x + y + z = 2$

$3x − 2y - z =3$

$x = 1, y = -1, z = 2$

## Operations with vectors

Think of vectors within a co-ordinate system as mentioned previously.

Think of the co-ordinates as scalars that describe but also allow us to manipulate vectors.

The basis vector for the $x$ axis is typically $\hat{i}$ and $\hat{j}$ for the $y$ axis.

Vector addition is _associative_ we can do it component by component.

> Formally, what this means is that if we have three vectors, r, s and, t, it doesn't matter whether we add r plus s and then add t, or whether we add r to s plus t, it doesn't matter where we put the bracket.

In the image below $\hat{i}$ and $\hat{j}$ are our _basis vectors_, things that define the space. Vector addition can be thought of as combining two scaled vectors of $\hat{i}$ and $\hat{j}$

![]({{ site.baseurl }}/images/linear_algebra/vector_addition.jpg){:width="400px"}

Mulitplication on vectors; $2r$ is 2 * the components of $r$

$$r = \begin{bmatrix}3 \\ 2\end{bmatrix}$$

$$2r = \begin{bmatrix}2*3 \\ 2*2\end{bmatrix} = \begin{bmatrix}6 \\ 4\end{bmatrix}$$ 

Vector subtraction; minus $r$ is not a shorter vector, that would be 0.5, rather it is in the oppositie direction. (Addition of negative one multiple the vector, ie vector + $(s * -1)$ -> this is a very handy way to think of subtraction)

![]({{ site.baseurl }}/images/linear_algebra/vector_operations.jpg){:width="400px"}

__Thinking of vectors as representing attributes of objects__

We don't have to think of vectors in a geometric space.

Vecotrs can represent information about an object. In this example, each vector will hold data relating to a house. Each row or co-ordinate,represents a separate piece of information; square metres, the number of bedrooms, bathrooms, and the price of a house.

Our vector operations still apply to these house objects, we can do operations on multiple houses, adding and multiplying (and introducing the concept of a negative house if there's such a thing)

![]({{ site.baseurl }}/images/linear_algebra/vectors_as_attributes.jpg){:width="400px"}

__Doing operations with vectors__

We have the following vectors;

![]({{ site.baseurl }}/images/linear_algebra/doingvectoroperations.png){:width="400px"}

$a$ = $\begin{bmatrix}2 \\ 2\end{bmatrix}$ 
$-b$ = $\begin{bmatrix}-1 \\ 2\end{bmatrix}$ 
$2c$ = $\begin{bmatrix}2 \\ 2\end{bmatrix}$ 
$d$ = $\begin{bmatrix}-1 \\ 2\end{bmatrix}$ 

__Calculate__

$b + e$ = $\begin{bmatrix}-1 \\ -1\end{bmatrix}$ 

$d - b$ = $\begin{bmatrix}-2 \\ 4\end{bmatrix}$  -> _think of it as $2d$, you're flipping $b$ because it's subtraction_

## Size of a vector

Addition and scaling by a number are two main vector operations and allow us define the mathematical properties that a vector has.

We can define a vector, say $r$, without any reference to any coordinate system. We can treat it as a geometric object, with just two properties, it's length (size) and its direction.

![]({{ site.baseurl }}/images/linear_algebra/vector_r.png){:width="200px"}

If we wanted to calculated the size, or the length of $r$, we could use a coordinate system with two axis that are orthogonal to each other.

![]({{ site.baseurl }}/images/linear_algebra/i_j_axis.png){:width="200px"}

$r = ai + bj$ -> a number of i, and b number of j

We assume $i$ and $j$ are of length one, denoted by $\hat{i}$ $\hat{j}$

In this case $r$ could often be written as a column vector, ignoring i and j;

$r = \begin{bmatrix}a \\ b\end{bmatrix}$ 

Length of $r$, also denoted as $\lVert r \rVert$ is giving by the hypotenuse. _Remember that thing? It's the longest side of a right-angled triangle_.

The size of a vector with two components is calculated using Pythagoras' theorem

$\lVert r \rVert = \sqrt{a^2 + b^2}$

![]({{ site.baseurl }}/images/linear_algebra/denoting_r_2d.png){:width="400px"}

__We define the size of a vector through the sums of the squares of its components__

In this case we've used two spatial directions (i and j) but __it doesn't matter if the different components of the vector are dimensions in space or even things that have different fiscal units__ like length, time and price.

In fact, this definition can be extended to __any number of dimensions__; the size of a vector is the square root of the sum of the squares of its components.

The size of vector $s = \begin{bmatrix}1 \\ 3 \\ 4 \\2\end{bmatrix}$ is $\lVert s \rVert = \sqrt{30}$

The size of a vector is equal to the square root of the dot product (we'll get to this) of the vector with itself;

$$\lVert r \rVert = \sqrt{r.r}$$

__Practice__

Let $a = \begin{bmatrix}3 \\ 0 \\ 4\end{bmatrix}$ and $b = \begin{bmatrix}0 \\ 5 \\ 12\end{bmatrix}$ 

Which is larger? $\lVert a + b \rVert$ or $\lVert a \rVert + \lVert b \rVert$

$\lVert a + b \rVert = \sqrt{3^2 + 5^2 + 16^2} = 17.029$

$\lVert a \rVert + \lVert b \rVert = 5 + 13 = 18$

This is known as _triangle inequality_

$$\lVert a + b \rVert \le \lVert a \rVert + \lVert b \rVert$$

[Short video from Khan Academy explaining the theorem](https://www.youtube.com/watch?v=KlKYvbigBqs)

Any one side of a triangle has to be less than the sum of the other two sides.

## Dot product, multiplying vectors

Say we have two vectors, $r$ and $s$, $r$ has the components $ri$ and $rj$, $s$ has the components $si$ and $sj$

![]({{ site.baseurl }}/images/linear_algebra/r_and_s.png){:width="300px"}

We define $r . s$ (the dot product) as multiplying the $i$ components and the $j$ components and then summing them.

$r.s = ri * si + rj * sj$

Another way of saying this is; the dot product of two vectors is the sum of their componentwise products;

![]({{ site.baseurl }}/images/linear_algebra/dotproduct2.png){:width="400px"}

```python
def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32 # 1 * 4 + 2 * 5 + 3 * 6
```

It's the length of the vector you'd get if you projected v onto w.

![]({{ site.baseurl }}/images/linear_algebra/dotproduct1.png){:width="400px"}

The dot product is therefore a scalar number not a vector. However, we get the vector of the dotted line above by multiplying $w$ by the dot product.

$r = \begin{bmatrix}3 \\ 2\end{bmatrix}$ $s = \begin{bmatrix}-1 \\ 2\end{bmatrix}$ dot product = 1

Two properties of the dot product;

(1) The dot product is __commutative__ ie `r.s == s.r`

(2) The dot product is __distributive over addition__ ie `r.(s + t) == r.s + r.t` - think in terms of unpacking

(3) The dot product is __associative over scalar multiplication__, similar to addition, here $a$ is a scalar number; `r.(as) == a(r.s)` we can just pull scalar numbers out

There is an interesting relationship between the size and the dot product of a vector.

If we dot a vector with its self $r.r$ we get the sum of the squares of its components (the square of it's size, it's modulus);

$$r.r = r_1 r_1 + r_2 r_2$$
$$  = r_1^2 + r_2^2$$
$$  = (\sqrt{r_1^2 + r_2^2})^2$$
$$  = \lVert r \rVert^2$$

So we can get the size of a vector by dotting it and taking the square root.

> The dot product for two $n$ component vectors $a, b = a_1b_1 + a_2b_2 + ... + a_nb_n$

The dot product of $r$ and $s$ below;

$r = \begin{bmatrix}-5\\3\\2\\8\end{bmatrix} s = \begin{bmatrix}1\\2\\-1\\0\end{bmatrix}$

$r.s = -5 + 6 + -2 + 0 = -1$

### Cosine & dot product

__Cosine rule__

If we have a triangle with sides a, b and c

$c^2 = a^2 + b^2 - 2ab * cos\theta$

Theta being the angle between a and b

The cosine rule, when combined with the dot product can tell us __the degree to which the two vectors are pointing in the same direction__.

![]({{ site.baseurl }}/images/linear_algebra/cosine_rule.png){:width="300px"}

First, translate out the cosine rule using vector notation;

![]({{ site.baseurl }}/images/linear_algebra/cosine_as_vector.png){:width="300px"}

We know that $\lVert r-s \rVert^2$ is equal to $(r-s).(r-s)$

$$(r-s).(r-s) = r.r \;\; r.-s \; -s.r \; -s.-s$$
$$  = \lVert r \rVert^2 \: -2s.r \;\; \lVert s \rVert^2 $$

Now we've multiplied out $\lVert r-s \rVert^2$

We can compare that to the right hand side.

$$\lVert r \rVert^2 \: -2s.r \;\; \lVert s \rVert^2 = \lVert r \rVert^2 + \lVert s \rVert^2 - 2\lVert r \rVert\lVert s \rVertcos\theta$$

Equivalent to;

$$ s.r = \lVert r \rVert\lVert s \rVertcos\theta $$

__The dot product takes the size of the vectors and multiples by the angle between them.__

This tells us the extent to which the two vectors go in the same direction.

> dot product of a vector is a scalar quantity describing only the magnitude of a particular vector

![]({{ site.baseurl }}/images/linear_algebra/dotproduct3.png){:width="400px"}

If the vectors are 90 degrees from one another
- they're orothognal
- (ie $\theta$ = 90),
- $cos 90 = 0$ 
- the dot product is 0

If the vectors are pointing in the same direction, 
- ie there is no angle 
- ($\theta$ = 0)
- $cos 0 = 1$ 
- the dot product equals the multiplication of the two sizes of the vectors (mod r $\lVert r \rVert$ and mod s $\lVert s \rVert$)
- a positive dot product tells us their moving in the same direction

If the vectors are pointing in opposite directions,
- ($\theta$ = 180), 
- $cos 180 = -1$ 
- the dot product equals the minus the multiplication of the two sizes of the vectors (mod r $\lVert r \rVert$ and mod s $\lVert s \rVert$)
- a negative dot product tells us their moving in opposite directions

From this, we've derived a property in the dot product;
$$ r.s = \lVert r \rVert\lVert s \rVert cos\theta$$

We can also use this formula to find the angle ($\theta$) between the two vectors

### Projection

Cast your mind back to sohcahtoa;

soh -> Sine = Opposite / Hypotenuse

cah -> Cosine = Adjacent / Hypotenuse

toa -> Tangent = Opposite / Adjacent

Here, the hypotenuse is the size of $s$

![]({{ site.baseurl }}/images/linear_algebra/projection.png){:width="200px"}

![]({{ site.baseurl }}/images/linear_algebra/projection3.png){:width="200px"}

Remember that the scalar projection is the size of the green vector.

If the angle between $s$ and $r$ is greater than $\pi/2$, the projection will also have a minus sign.

We can substitute this in with our dot product

$$ r.s = \lVert r \rVert\lVert s \rVert cos\theta$$

$\lVert s \rVert cos\theta$ is the adjacent side (dotted line in images above)

Think of the projection as a light coming down from $s$ at a right angle and the shadow cast onto $r$. If $s$ and $r$ are at 90 degrees, it would be 0.

The projection of $s$ onto $r$ is not the same as $r$ onto $s$ - the light will be pointing at different angles.

The dot product gives us the projection multiplied by the size of r $\lVert r \rVert$

![]({{ site.baseurl }}/images/linear_algebra/projection2.png){:width="200px"}

We can get the adjacent size by diving the dot product by mod r

$$ \frac{r.s}{\lVert r \rVert} = \lVert s \rVert cos\theta$$

Remember, $r.s$ is a number and $\lVert r \rVert$ is a number so you get a number, known as the `scalar projection`

The dot product is also known as the projection product. It takes the projection of one vector onto another.

We can do projection in any number of dimensions. Consider two vectors with three components

$r = \begin{bmatrix}3\\-4\\0\end{bmatrix} s = \begin{bmatrix}10\\5\\-6\end{bmatrix}$

The scalar projection of $s$ on $r$;

$$ proj_rs = \frac{s.r}{\lVert r \rVert}$$

$s.r = 10;\;\; r.r = 5$

__The vector projection__ allows us to encode something about which direction $r$ was going, into the `scalar projection`

Vector projection;
$$ r\; \frac{r.s}{\lVert r \rVert \lVert r \rVert} = r\; \frac{r.s}{r.r}$$

We've take the scalar projection (how much $s$ goes along $r$ or $\frac{r.s}{\lVert r \rVert}$ and multiplied it by $r$ divided by its length or $r\; \frac{1}{\lVert r \rVert}$ (this produces a vector going the direction of $r$ but normalised to have a length 1) 

The vector projection is a number multiplied by a unit vector (that goes in the direction of $r$)

Taking our example from above, given the the scalar projection is 2, the vector projection of $s$ onto $r$

$r = \begin{bmatrix}3\\-4\\0\end{bmatrix} s = \begin{bmatrix}10\\5\\-6\end{bmatrix}$

The scalar projection of $s$ on $r$;

$$ proj_rs = \frac{s.r}{\lVert r \rVert} = 2$$

$s.r = 10;\;\; r.r = 5$

The vector projection is;

$$\frac{s.r}{r.r}r$$

So you can multiple $r$ by the scalar projection and then divide by $r$

$\begin{bmatrix}3\\-4\\0\end{bmatrix} *2 = \begin{bmatrix}6\\-8\\0\end{bmatrix}$

normalised by length r = $\begin{bmatrix}6/5\\-8/5\\0\end{bmatrix}$

## Basis, the coordinate system in which our vectors exists

The coordinate system is what we use to describe space. These are our basis vectors.

We describe our vectors in terms of our basis vectors.

Any time we describe vectors numerically it implies implicitly on what basis vectors we're using.

$r$ described in terms of $e_1$ and $e_2$ (hat -> $\hat{e}$ represents they are of unit length)

In a 2D plane we can describe all points with different combinations of our two basis vectors (_as long as linearly independent_ i.e. not pointing in the same direction).

The "span" of $\hat{e_1}$ and $\hat{e_2}$ is the set of all their linear combinations; $a\hat{e_1} + b\hat{e_2}$

![]({{ site.baseurl }}/images/linear_algebra/r_intermsofe.png){:width="400px"}

$r$ exists independently of the basis vectors. It takes us from an origin to another point, but we use the coordinate system to describe $r$.

If we want to do a __transformation of axes__ i.e. change the coordinate system we use to describe $r$, __if__ the vectors are at a 90 degree angle from one another we can use the _dot product_, otherwise we need to use a matrix.

Here, we know $b$ in terms of $e$ and the vectors are at a 90 degreee angle to one another (orthogonal), so we can work out $r$ in terms of $b$ using the dot product.

You check that two vectors are orthogonal by taking the dot product, multiply two vectors and divide by their length;

$$ cos\theta = \frac{b_1.b_2}{\lVert b_1 \rVert\lVert b_2 \rVert}$$

If the dot product is 0, $cos\theta$ is 0 and they're at 90 degrees to one another.

Using the example below;

$$b_1.b_2 = (2*-2) + (1*4) = 0$$

![]({{ site.baseurl }}/images/linear_algebra/axistransformation1.png){:width="400px"}

You can project $r$ down onto $b_2$ and calculate the vector projection. This will tell you "how much of that axis ($b_2$) you need". You then do the same for $b_1$

The sum of those two vector projections, give you $r$ in terms of $b$ or $r_b$.

The vector projection for $b_1$;

$$r_e = \begin{bmatrix}3\\4\end{bmatrix}\;\; b_1 = \begin{bmatrix}2\\1\end{bmatrix}$$

$$\frac{r_e.b_1}{\lVert b_1 \rVert^2} = \frac{(3*2)+(4*1)}{2^2 + 1^2} = \frac{10}{5} = 2$$

And the same for the other axis

$$b_2 = \begin{bmatrix}-2\\4\end{bmatrix}$$

$$\frac{r_e.b_2}{\lVert b_2 \rVert^2} = \frac{(3*-2)+(4*4)}{-2^2 + 4^2} = \frac{10}{20} = 1/2$$

$$r_b = \begin{bmatrix}2\\1/2\end{bmatrix}$$

We can verify by multiplying our $b$s in terms of $e$ by the scalar projections (normalising) to product $r_e$

$$b_1 = \begin{bmatrix}2\\1\end{bmatrix} *2 = \begin{bmatrix}4\\2\end{bmatrix}$$

$$b_2 = \begin{bmatrix}-2\\4\end{bmatrix} *1/2 = \begin{bmatrix}-1\\2\end{bmatrix}$$

$$\begin{bmatrix}4\\2\end{bmatrix} + \begin{bmatrix}-1\\2\end{bmatrix} = \begin{bmatrix}3\\4\end{bmatrix} $$

We can re-describe $r$ using a new set of basis vectors.

----

__Practice__

(1)

Given the following vectors are written in the standard basis, represent v in terms of b;

$$v = \begin{bmatrix}5\\-1\end{bmatrix}\;\; b_1 = \begin{bmatrix}1\\1\end{bmatrix}\;\; b_2 = \begin{bmatrix}1\\-1\end{bmatrix}$$

```python
vb1 = ((5 * 1) + (-1 * 1)) / (1**2 + 1**2) # 2

vb2 = ((5 * 1) + (-1 * -1)) / (1**2 + pow(-1, 2)) # 3
```

$$v_b = \begin{bmatrix}2\\3\end{bmatrix}$$

(2)

$$v = \begin{bmatrix}10\\-5\end{bmatrix}\;\; b_1 = \begin{bmatrix}3\\4\end{bmatrix}\;\; b_2 = \begin{bmatrix}4\\-3\end{bmatrix}$$

$$v_b = \begin{bmatrix}2/5\\11/5\end{bmatrix}$$

(3)

$$v = \begin{bmatrix}2\\2\end{bmatrix}\;\; b_1 = \begin{bmatrix}-3\\1\end{bmatrix}\;\; b_2 = \begin{bmatrix}1\\3\end{bmatrix}$$

$$v_b = \begin{bmatrix}-2/5\\4/5\end{bmatrix}$$

(4)

$$v = \begin{bmatrix}1\\1\\1\end{bmatrix}\;\; b_1 = \begin{bmatrix}2\\1\\0\end{bmatrix}\;\; b_2 = \begin{bmatrix}1\\-2\\-1\end{bmatrix}\;\; b_3 = \begin{bmatrix}-1\\2\\5\end{bmatrix}$$

$$v_b = \begin{bmatrix}3/5\\-1/3\\-2/15\end{bmatrix}$$

(5)

$$v = \begin{bmatrix}1\\1\\2\\3\end{bmatrix}\;\; b_1 = \begin{bmatrix}1\\0\\0\\0\end{bmatrix}\;\; b_2 = \begin{bmatrix}0\\2\\-1\\0\end{bmatrix}\;\; b_3 = \begin{bmatrix}0\\1\\2\\0\end{bmatrix}\;\;
b_4 = \begin{bmatrix}0\\0\\0\\3\end{bmatrix}$$

$$v_b = \begin{bmatrix}1\\0\\1\\1\end{bmatrix}$$

****

### Defining basis, vector space and linear independence

A basis is a set of $n$ vectors that;

- Are not linear combinations of each other
  <br> and are therefore linearly independent
- Span the space
- Our space is therefore n-dimensional

$b_3$ is not a valid basis vector. Because I can take some combination of $b_2$ and $b_1$ to get $b_3$

We cannot write one of the vectors as a linear combination of the others

ie: $b_3 = a_1 b_1 + a_2 b_2$

"It lies in the same plane as $b_1$ and $b_2$"

Formula required to show linear independence;

![]({{ site.baseurl }}/images/linear_algebra/basis_linearindependence.png){:width="400px"}

Linearly __dependent__;

$a = \begin{bmatrix}1\\1\end{bmatrix}\;\; b = \begin{bmatrix}2\\2\end{bmatrix}$

as;

$a = \frac{1}{2}b$

Similarly, $\mathbf{a} = q_1\mathbf{b} + q_2\mathbf{c}$

![]({{ site.baseurl }}/images/linear_algebra/basis_linearindependence2.png){:width="400px"}

Find $q_1, q_2$

$\begin{bmatrix}2\\2\end{bmatrix}\;\; = q_1 \begin{bmatrix}1\\-2\end{bmatrix} + q_2 \begin{bmatrix}-1\\0\end{bmatrix}$

$q_1 = -1,\;\; q_2 = -3$

Linearly __independent__; one is not a scalar multiple of the other

$a = \begin{bmatrix}1\\1\end{bmatrix}\;\; b = \begin{bmatrix}2\\1\end{bmatrix}$

Basis vectors do not need to be;
- Of unit length (length one)
- Orthogonal
- Though it's much if they are both of these things

Linearly __independent__

$a = \begin{bmatrix}1\\0\\0\end{bmatrix}\;\; b = \begin{bmatrix}1\\1\\0\end{bmatrix}\;\; c = \begin{bmatrix}1\\0\\1\end{bmatrix}$

Easy to tell, as you need all three vectors to cover each element. Whereas in the following example, you do not;

Linearly __dependent__

$a = \begin{bmatrix}1\\2\\0\end{bmatrix}\;\; b = \begin{bmatrix}-2\\1\\3\end{bmatrix}\;\; c = \begin{bmatrix}4\\3\\-3\end{bmatrix}$

$c = 2a - b$

__What happens when we map from one basis to another?__

The original grid projects down onto the new grid.

Though it will potentially have different values on that grid, the projection keeps the grid being evenly spaced. 

Therefore, any mapping we do from one set of basis vectors, from one coordinate system to another, keeps the vector space being a regularly spaced grid. Ensuring, our original vector rules of vector addition and multiplication by a scalar still work.

It doesn't warp or fold space which is what the linear bit in linear algebra means. Things might be stretched or rotated or inverted, but everything remains evenly spaced and linear combinations still work.

![]({{ site.baseurl }}/images/linear_algebra/axistransformation2.png){:width="400px"}


### Applications of changing basis

Changing the basis is what we're doing when we do linear regression.

We're working out the distance of points from a vector (distance least squared).

We use the dot-product to do the projection to map the data from the x-y space onto the space of the line.

There's some theoretical disputes, whether the distance should be measured straight (y-axis) or orthognally (the angle).

![]({{ site.baseurl }}/images/linear_algebra/applicationofchangbasis.png){:width="400px"}

In a neural network for face recognition, the goal of the learning process is going to be to somehow derive a set of basis vectors that extract the most information-rich features of the faces.

----

__Practice__

__Q1__

A ship travels with the velcoity given by $\begin{bmatrix}1\\2\end{bmatrix}$, with current flowing in the direction given by $\begin{bmatrix}1\\1\end{bmatrix}$ with respect to some co-ordinate axes.

What is the velocity of the ship in the direction of the current?

- vector projection of the velocity of the ship, onto the velocity of the current

$\frac{ship . current}{current . current} * current$

$\frac{\begin{bmatrix}1\\2\end{bmatrix} . \begin{bmatrix}1\\1\end{bmatrix}} {\begin{bmatrix}1\\1\end{bmatrix} . \begin{bmatrix}1\\1\end{bmatrix}} * \begin{bmatrix}1\\1\end{bmatrix}$

$\begin{bmatrix}3/2\\3/2\end{bmatrix}$

__Q2__

A ball travels with the velocity given by $\begin{bmatrix}2\\1\end{bmatrix}$, with wind blowing in the direction given by $\begin{bmatrix}3\\-4\end{bmatrix}$ with respect to some co-ordinate axes.

What is the size of the velocity of the ball in the direction of the wind?

This is the scalar projection of the velocity of the ball onto the velocity of the wind.

If you were to draw a straight line from the co-ordinate of the ball onto the vector of the wind, what is the size of that vector in terms of the wind vector.

![]({{ site.baseurl }}/images/linear_algebra/projection3.png){:width="200px"}

Remember that the scalar projection is the size of the green vector.

$\frac{\begin{bmatrix}2\\1\end{bmatrix} . \begin{bmatrix}3\\-4\end{bmatrix}} {\sqrt{3^2 + -4^2}}$

$\frac{2}{5}$

__Q3__

Given vectors $v = \begin{bmatrix}-4\\-3\\8\end{bmatrix}\; b_1 = \begin{bmatrix}1\\2\\3\end{bmatrix}\; b_2 = \begin{bmatrix}-2\\1\\0\end{bmatrix}\; b_3 = \begin{bmatrix}-3\\-6\\-5\end{bmatrix}$

All written in the standard basis, what is $v$ in the basis defined by $b_1, b_2, b_3$? -> they are all pairwise orthogonal to one another.

_Answer_ What manipulation do you need to do in order to produce $v$. Simply add them together, so one of each vector.

This is a change of basis in 3 dimensions.

$\begin{bmatrix}1\\1\\1\end{bmatrix}$

__Q4__

Are the following vectors linearly independent?

$a = \begin{bmatrix}1\\2\\-1\end{bmatrix}\; b = \begin{bmatrix}3\\-4\\5\end{bmatrix}\; c = \begin{bmatrix}1\\-8\\7\end{bmatrix}$

No, $b = 2a + c$

__Q5__

At 12:00 pm, a spaceship is at position $\begin{bmatrix}3\\2\\4\end{bmatrix} km$ away form the origin with respect to some 3 dimensional co-ordinate system. The ship is travelling with velocity $\begin{bmatrix}3\\2\\4\end{bmatrix} km/h$. What is the location of the spaceship after 2 hours have passed?

_Answer_ Multiply the velocity by 2 and then add it to the current position to get the new position.

$\begin{bmatrix}1\\6\\-2\end{bmatrix}$

## Introduction to matrices

> Unfortunately, no one can be told what the Matrix is. You have to see if for yourself - _Morpheus_

Morpheus may have a point. Like the proverbial fish who has no idea what water is, we are swimming in matrices.

> Matrices are everywhere; anything that can be put in an Excel spreadsheet is a matrix, and language and pictures can be represented as matrices as well. [_fast ai_](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md)

A matrix is a two-dimensional collection of numbers, a list of lists. If a matrix has $n$ rows and $k$ columns, we refer to it as an $n \times k matrix$. We can think of each column in a matrix as a vector of length $n$.

When working with matrices to represent data, we need to think of them slightly differently to tabular data structures. However, they can to some extent both follow the [tidy data](https://vita.had.co.nz/papers/tidy-data.pdf) philosophy where each column represents an attribute and each row a record.

Matrices can also be used to represent binary relationships, where if `A[i][j] = 1` then nodes $i$ and $j$ are connected (see _hot encoding_).

Matrices can also be thought of as objects that rotate and stretch vectors. They also help us solve simultaneous equations.

$$2a + 3b = 8$$
$$10a + 1b = 13$$

Can be rewritten as;

$$\begin{pmatrix} 2a + 3b \\ 10a + 1b \end{pmatrix} = \begin{pmatrix} 8 \\ 13 \end{pmatrix}$$

Or

$$\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} 8 \\ 13 \end{pmatrix}$$

The matrix $\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix}$ acts on the vector $\begin{bmatrix}a\\b\end{bmatrix}$

So we then ask, what vector transforms to produce $\begin{pmatrix} 8 \\ 13 \end{pmatrix}$

This is the heart of linear algebra.

> Now we can see what we mean now by the term linear algebra.
> 
> Linear algebra is linear, because it takes input values, our a and b, and multiplies them by constants. So everything is linear. And it's algebra, that is it's a notation describing mathematical objects and a system of manipulating those notations. 
> 
> So linear algebra is a mathematical system for manipulating vectors in the spaces described by vectors. 
> 
> So this is interesting. There seems to be some kind of deep connection between simultaneous equations, these things called matrices, the vectors. And it turns that the key to solving simultaneous equation problems is appreciating how vectors are transformed by matrices, which is the heart of linear algebra.

We can multiply the matrix $\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix}$  by some basis vectors, for x we have $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and for y we have $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ 

When multiplying vectors;

$$\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} * \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2*1 + 3*0 \\ 10*1 + 1*0 \end{pmatrix} = \begin{pmatrix} 2 \\ 10 \end{pmatrix}$$

$$\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} * \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 2*0 + 3*1 \\ 10*0 + 1*1 \end{pmatrix} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$$

So the matrix $\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix}$ is transforming our basis vectors, it's a function that operates on input vectors and gives us output vectors.

A set of simultaneous equations is asking what vector I need in order to get a transformed product at the position 8 13.

### Linear transformations

Transformation is a another term for function $f(x)$, it takes an input and produces an output.

In linear algebra a matrix transformation takes in a vector and produces another vectors. Transformation suggests movement, again thinking geometrically.

We can think of every corresponding input vector within a space, moving to its corresponding output vectors.

Think of vectors as points (i.e. where the arrow points) and think of that point moving to some other point.

![]({{ site.baseurl }}/images/linear_algebra/lineartransformations1.png){:width="400"}

We have the ability to move around all the points in space. Though the types of transformations are limited to `linear` transformations.

Visually, what this means is that all lines in our original coordinate system must remain lines, and our origin remains fixed.

![]({{ site.baseurl }}/images/linear_algebra/lineartransformations2.png){:width="400"}

It's not just the vertical and horizontal lines that must remain lines, but all angles, so diagonals as well. Grid lines remain parallel and evenly spaces.

The implications of this are important.

A vector, say $v = \begin{bmatrix} -1\\2 \end{bmatrix}$

We know is actually, $\begin{bmatrix} -1\hat{i}\\2 \hat{j}\end{bmatrix}$

When do we the transformation, we simply need to understand what happens $\hat{i}$ and $\hat{j}$.

$v$ will still be the same linear combination of $\hat{i}$ and $\hat{j}$.

transformed $v$ = -1(transformed $\hat{i}$) + 2(transformed $\hat{j}$)

In a 2 dimensional matrix, we need just four numbers to know where any vector will now land. The coordinates for $\hat{i}$ and $\hat{j}$.

![]({{ site.baseurl }}/images/linear_algebra/lineartransformations3.png){:width="400"}

Any vector can then be translated from the old basis vectors to the new transformation.

![]({{ site.baseurl }}/images/linear_algebra/lineartransformations4.png){:width="400"}

`in matrix multiplication, remember; rows times cols`

The intuition here is to think of the two columns in the matrix as where your basis vectors end up, and your vector as a linear combination of your new basis vectors.

A matrix is therefore a transformation of space.

### How matrices transform space

Space changes include;

- stretches,
- inversions,
- mirrors,
- shears,
- rotations

> We can think of a matrix multiplication being the multiplication of the vector sum of the transformed basis vectors.

The identity matrix $(I)$, the matrix that does nothing. It is composed of the basis vectors.

$$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

The "Identity Matrix" is the matrix equivalent of the number "1". It is "square" (has same number of rows as columns)

Gives us an axis of 1 x, 1 y --> $\begin{bmatrix}x\\y\end{bmatrix}$

The matrix;
$$\begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$

Would expand our basis vectors by 3x and 2y.

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation1.png){:width="200"}

 A fraction would shrink space.

 A negative reverses the space. "Changing the sense of the co-ordinate system"

$$\begin{pmatrix} -1 & 0 \\ 0 & 2 \end{pmatrix}$$

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation2.png){:width="200"}

Two negatives will invert everything. Inversion.

You can have a matrix that is akin to having a mirror, where it shifts both axses

$$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation3.png){:width="400"}

The following matrix acts as a vertical mirror plane $\begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$

The following matrix acts as a horizontal mirror plane $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

You can also have `shears`

> In plane geometry, a shear mapping is a linear map that displaces each point in fixed direction, by an amount proportional to its signed distance from the line that is parallel to that direction and goes through the origin. This type of mapping is also called shear transformation, transvection, or just shearing.

For instance, keeping $\hat{e}_1$ in place but transforming $\hat{e}_2$ to $\prime{e}_2$ (e-prime)

$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ to $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$

Creates a parallelogram;

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation4.png){:width="400"}

A 90 degree anticlockwise rotation will look like;

$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ to $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation5.png){:width="400"}

A general expression for a rotation in 2d can be written as;

$\begin{bmatrix} cos\theta & sin\theta \\ -sin\theta & cos\theta \end{bmatrix}$

Rotations are less relevant in most data science applications of linear algebra. But image classification and facial recognition would use this transformations in order to get all the images in a certain state before analysing them - ie remove the camera angles and make the faces portrait.

### Combining matrix transformations

If you apply one transformation (a rotation), and then another (a shear), you live get a linear transformation that is distinct from the rotation and shear.

Applying multiple transformations forms a composition. The composition is the product of multiple matrices.

The composition can be described by its own matrix.

Like function notation, we read from inside out $f(g(x))$

![]({{ site.baseurl }}/images/linear_algebra/composition1.png){:width="400"}

You can see why we do `rows times cols`. 

- The matrix first applied is where the new basis vectors land. 
- $\hat{i}$ and $\hat{j}$ are represented as columns in the matrix on the right. 
- We then need to multiply each vector (column) by the new matrix.

The image below gives us the first column of our composition.

![]({{ site.baseurl }}/images/linear_algebra/composition2.png){:width="400"}

The intuition is to think of applying one matrix transformation after another. This is why the order in which the transformations are applied matters. To rotate and then shear has a different effect than if you shear and then rotate.

Matrix multiplication is _associative_ $AB(C) \equal A(BC)$, because this doesn't change the order in which the transformations are applied.

If we take the basis vectors
$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$

$A_1$ represents a 90 degree clockwise rotation $\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$

$A_2$ represents a mirror along the vertical plane (_ie_ flips the x axis, with no change to y) $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$

We can calculate the result of applying the two transformation $A_2(A_1 r)$ by first applying $A_1$ to our basis vectors and then applying $A_2$ to the result. 

$$A_2 A_1 = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$

We can multiply all possible rows and columns, without having to draw it out. 

- Top left value = the first row of $A_2$ and multiply it by the first column in $A_1$,
- Bottom left value = second row, first column
- Top right = first row, second column
- Bottom right = second row, second column 

$$ = \begin{bmatrix} (-1 * 0) + (0 * -1) & (-1 * -1) + (0*0)\\ (-1 * 1) + (0 * 0) & (0 * 1) + (1 * 0) \end{bmatrix}$$

$$ = \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix}$$

The order of transformations matter, doing $A_2$ and then $A_1$ gives us a different result.

$$A_1 A_2 = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$A_1 A_2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

__Matirx multiplication isn't commutative__

We can do them in any order, meaning they are associative. $A_3(A_2 A_1)$ is the same as $(A_3 A_2) A_1$ but we cannot interchange the order.

### Practice matrix transformations

Matrices make transformations on vectors, potentially changing their magnitude and direction.

If we have two unit vectors (in orange) and another vector, $r = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$ (in pink), before any transformations - these look like this:

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation6.png){:width="200"}

The matrix, $A = \begin{pmatrix} 1/2 & -1 \\ 0 & 3/4 \end{pmatrix}$ will transform them;

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation7.png){:width="200"}

__Q1__

r prime, or $A$ applied to $r$ can be written as;

$Ar = \begin{pmatrix} 1/2 & -1 \\ 0 & 3/4 \end{pmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix}$

$\prime{r} = \begin{bmatrix} -1/2 \\ 3/2 \end{bmatrix}$

`(1/2 * 3) + (-1 * 2) = -1/2`

`(0 * 3) + (3/4 * 2) = 3/2`

__Q2__

$s = A \begin{bmatrix} -2 \\ 4 \end{bmatrix}$

`(1/2 * -2) + (-1 * 4) = -5`

`(0 * -2) + (3/4 * 4) = 3`

__Q3__

$M = \begin{bmatrix} -1/2 & 1/2 \\ 1/2 & 1/2 \end{bmatrix}$

Thoughts; 
- Everything gets smaller.
- Top lefts inversion should flip the horizontal axis.
- initially thought that because diagonals did not align it would be a parallelogram rather than a square, but not the case because it is a sheer and scale transformation
- The axis have been rotated and flipped

Using unit vectors $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$

top left -0.5
bottom left 0.5
top right 0.5
bottom right 0.5

Best corresponds to;

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation8.png){:width="200"}

__Q4__

Quick hack - anticlockwise rotation requires top right to be negative, all else positive.

__Q5__

$M = \begin{bmatrix} 1 & 0 \\ 0 & 8 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ -1/2 & 1 \end{bmatrix}$

`top left 1*1 + 0*-1/2 = 1`

`bottom left 0*1 + 8*-1/2 = -4` 

`top right 1*0 + 0x1 = 0`

`bottom right 0x0 + 8x1 = 8`

### Matrix Inverses

__The apples and bananas problem__

I walked into a shop and bough 2 apples and 3 bananas for a price of £8. Another day I bought 10 apples and 1 banana for a price of £13.

This is a matrix (fruits $A$) * a vector (prices $r$) and produces an output vector (purchase $s$).

$A$ operates on $r$ and gives $s$

$$ \begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} 8 \\ 13 \end{bmatrix}$$

$A^{-1}$, the inverse of $A$ does exact opposite of what $A$ does.

By multiplying $A$ by $A^{-1}$, we reverse whatever $A$ does and we get the identity matrix $I$, the matrix that does nothing.

$Ar = s$ becomes $A^{-1} Ar = A^{-1} s$

$A^{-1} A$ is the identity matrix ($A^{-1} A = I$), the matrix that does nothing, so you just have $r$

$r = A^{-1} s$

### Linear algebra to solve systems of equations

All of sudden we've moved away from planning with geometry and vector spaces and back to linear equations.

Systems of equations are a list of variables (things you don't know) and a list of equations relating to them.

This is one of the core applications of linear algebra, it allows us to solve "systems of equations".

The following system of equations;

$$4x_1 = 5x_2 = - 13$$
$$-2x_1 + 3x_2 = 9$$

In matrix notation, we can write the system more compactly;

$Ax = b$ with;

$$A = \begin{bmatrix} 4 & -5 \\ -2 & 3 \end{bmatrix}, b = \begin{bmatrix} -13 \\ 9 \end{bmatrix}$$

If the equations can be solved using linear algebra then (1) the variable in each equation is being scaled by some constant and (2) those variables are being added to each other in the equation. There are no exponents or multiplying variables together.

Arranging our systems of equations like this, sheds some geometric light on the problem;

![]({{ site.baseurl }}/images/linear_algebra/systemsofequations1.png){:width="400"}

We're looking for a vector $\bf{x}$, that after applying the transformation $A$, lands on $\bf{v}$

If $A$ manipulated $\bf{x}$ such that it rotated 90 degrees counter-clockwise, then we need to find the matrix that reverses that transformation, i.e. a 90 degree clockwise rotation.

If $A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, A^{-1} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$

If you first apply $A$, then $A^{-1}$ you end up where you started.

![]({{ site.baseurl }}/images/linear_algebra/systemsofequations3.png){:width="200"}

$A^{-1}A$ gets you back to where you started; multiplied out it gives you the identity matrix, the equivalent of multiplying by 1 in matrix multiplication.

Therefore;

![]({{ site.baseurl }}/images/linear_algebra/systemsofequations3.png){:width="400"}

If the determinant of $A$ is zero, it has manipulated space into a lower dimension, as a result we cannot use it's inverse to understand what matrix was applied onto $\bf{x}$ to produce $\bf{v}$.

The equation can be solved, but only with the vector $\bf{v}$ lives on the new dimension (in the case of 3D to 2D, the vector must live on that line).

![]({{ site.baseurl }}/images/linear_algebra/systemsofequations4.png){:width="300"}

When the output of a transformation is a line, i.e. it's one-dimensional, we say the transformation has a rank of 1. If all vectors land on a two-dimensional place, the transformation has a rank of 2, and so on.

The set of outputs for the transformation matrix is called the _column space_. The columns tell you where your vectors land, and the span of the columns gives you all possible outputs.

![]({{ site.baseurl }}/images/linear_algebra/systemsofequations5.png){:width="300"}

The rank is the number of dimensions in the column space. If a rank mantains it's number of dimensions, then it is regarded as "_full rank_".

When a matrix has full rank, only the point on the origin is the original origin. When a matrix loses dimensions, many points will now fall on the origin. Think of a line coming down to a single point. Lots of lines now sit at the origin. The set of vectors that land on the origin is called the "_null space_" or the "_kernel_" of your matrix. The space of all vectors that become null.

### Solving systems of equations

__Elimination__

With a more complex example. We can simplify the problem by removing row one from the other two rows

$$ \begin{pmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{pmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 15 \\ 21 \\ 13 \end{bmatrix}$$

Removing row one gives us;

$$ \begin{pmatrix} 1 & 1 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & -1 \end{pmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 15 \\ 6 \\ -2 \end{bmatrix}$$

We've now solved $c$, $-c = -2$, $c = 2$

A matrix like this, where everything below the body diagonal is 0 is known as a triangular matrix.

The matrix has been reduced to what's known as `Echelon form`. All the numbers below the leading diagonal is zero.

We can now use back substitution. Using the value of $c$  nad plugging back into the first two rows.

Removing $c$ from the rows we get;

$$ \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 9 \\ 4 \\ 2 \end{bmatrix}$$

Then getting to the final solution we remove $b$ from row 1;

$$ \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 5 \\ 4 \\ 2 \end{bmatrix}$$

However, we found $r$ for a specific output of $s$, we didn't compute the general. Calculating the inverse allows us to ensure we have the general case. We could find $r$ for any $s$.

Elimination and back substitution are extremely computational efficient.

__Finding the inverse matrix__

$A B = I$ where $B$ is the inverse of $A$, $B = A^{-1}$. You can apply the inverse to the right or the left, it's commutative.

$$ A = \begin{pmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{pmatrix} B = \begin{pmatrix} b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23} \\ b_{31} & b_{32} & b_{33} \end{pmatrix}$$

The identity matrix is $I = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$

You could perform back substitution, by taking each column in the inverse matrix and use the output of the identity matrix.

$$ \begin{pmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{pmatrix} * \begin{pmatrix} b_{11}\\ b_{21}\\ b_{31} \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$$

However, we can use elimination to do it more efficiently.

Subtracting the first row from rows two and three.

$$A* \begin{pmatrix} 1 & 1 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & -1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ -1 & 0 & 1 \end{pmatrix}$$

If we multiply the bottom row by -1 we can get the matrix in echelon form;

$$A* \begin{pmatrix} 1 & 1 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ 1 & 0 & -1 \end{pmatrix}$$

Then back substitute, make the third column 0 in both rows one and two.

Take the bottom row and subtract it from the second row and multiply by three and subtract from row one;

$$A* \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} -2 & 0 & 3 \\ -2 & 1 & 1 \\ 1 & 0 & -1 \end{pmatrix}$$

Then back substitute b (the middle column) from the first row;

$$A* \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} -2 & -1 & 2 \\ -2 & 1 & 1 \\ 1 & 0 & -1 \end{pmatrix}$$

This provides us with the inverse matrix.

$$ \begin{pmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{pmatrix} * \begin{pmatrix} -2 & -1 & 2 \\ -2 & 1 & 1 \\ 1 & 0 & -1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

Computationally this approach is much easier, particularly when the dimensions increase. There are computationally faster methods to do a decomposition process.

Most software packages come with a solver function that we simply have to call on the matrix, and it will apply the most efficient process `inv(A)`.

The method above is a general method that we could write ourselves.

### Practice solving linear equations using the inverse matrix

__Q1__

You go to the shops on Monday and buy 1 apple, 1 banana, and 1 carrot; the whole transaction totals €15. On Tuesday you buy 3 apples, 2 bananas, 1 carrot, all for €28. Then on Wednesday 2 apples, 1 banana, 2 carrots, for €23.

$$ A * \begin{bmatrix} a \\ b \\ c \end{bmatrix} =\begin{pmatrix} sMon \\ sTue \\ sWed \end{pmatrix}$$

```python
A = [[1, 1, 1],
     [3, 2, 1],
     [2, 1, 2]]

s = [15, 28, 23]
```

__Q2__

Given another system, $Br = t$

![]({{ site.baseurl }}/images/linear_algebra/solvinglineq_inverse1.png){:width="450"}

We need to time the first row by three, then minus that from the second row. This will ensure the first value of the row is 0.

Then we need to multiply by minus 2 to ensure the second value of the row is 1.

$2{''} = [2{'} - (3*1{'})] * -2$

__Q3__

From our previous question;

$$ \begin{pmatrix} 1 & 3/2 & 1/2 \\ 0 & 1 & 1 \\ 2 & 8 & 13 \end{pmatrix} * \begin{bmatrix} a\\ b\\ c\end{bmatrix} = \begin{pmatrix} 9/4 \\ -1/2 \\ 2 \end{pmatrix}$$

Fix row 3 to be a linear combination of the other two and provide a matrix in echelon form.

```python

row_3 = [2, 8, 13] = 2

# Multiply row 1 by 2
[1, 3/2, 1/2] * 2 = [2, 3, 1]
9/4 * 2 = 4.5

# Subtract from 3
[2, 8, 13] - [2, 3, 1] = [0, 5, 12]
2 - -2.5 = -2.5

# Multiply row 2 by 5
[0, 1, 1] * 5 = [0, 5, 5]
-0.5 * 5 = -2.5

# Subtract from 3'
[0, 5, 12] - [0, 5, 5] = [0, 0, 7]
-2.5 - -2.5 = 0

# Divide by 7
[0, 0, 7] / 7 = [0, 0, 1]
0/7 = 0
```

$$ \begin{pmatrix} 1 & 3/2 & 1/2 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix} * \begin{bmatrix} a\\ b\\ c\end{bmatrix} = \begin{pmatrix} 9/4 \\ -1/2 \\ 0 \end{pmatrix}$$

We can then compute a, b, c

b = -1/2 (ignore c as 0)

a = 9/4 - (-1/2 * 1.5) = 3 (multiply row 2 by 1.5 then remove it, ignore row 3 as 0)

$r = \begin{bmatrix} 3 \\ -1/2 \\ 0 \end{bmatrix}$

__Q5/6__

Returning to Q1, convert the system to echelon form

```python
A = [[1, 1, 1],
     [3, 2, 1],
     [2, 1, 2]]

s = [15, 28, 23]

# Answer;
# Row 1 * 2
[2, 2, 2] , 30

# Sub from Row 3
[0, -1, 0] , -7
# Multiply by -1
R3` = [0, 1, 0], 7

# Add R2 to R3`
[3, 3, 1], 35

# Row 1 * 3 and sub from R3``
[0, 0, -2], -10
# Multiply by -0.5
R3` = [0, 0, 1], 5

# Row 1 * 3 and sub from R2
[0, -1, -2], -17
# Multiply by -1
R2` = [0, 1, 2], 17

A = [[ 1 , 1, 1],
     [ 0,  1 , 2],
     [ 0,  0 , 1]]
s = [15, 17, 5]

# Price of individual elements;
s = [3, 7, 5]     
```

__Q7__

Find the inverse of the matrix you used in Question 1

```python
# Answer;
Ainv = [[-1.5,  0.5,  0.5],
       [ 2.0,  0.0, -1.0],
       [ 0.5, -0.5,  0.5]]
```

__Q8__

In practice, for larger systems, one never solves a linear system by hand as there are software packages that can do this for you - such as numpy in Python.

```python
import numpy as np

A = [[1, 1, 3],
     [1, 2, 4],
     [1, 1, 2]]

Ainv = np.linalg.inv(A)
```
In general, one shouldn't calculate the inverse of a matrix unless absolutely necessary. It is more computationally efficient to solve the linear algebra system if that is all you need.

Numpy can also do this for you.

```python
import numpy as np
A = [[4, 6, 2],
     [3, 4, 1],
     [2, 8, 13]]

s = [9, 7, 2]

r = np.linalg.solve(A, s)
```

### Determinants

> The determinant of a linear transformation measures how much areas/volumes change during the transformation.

The matrix scales space. Creating $e'_1$ and $e'_2$ from our original basis vectors.

The space has been expanded a factor of $a$ horizontally and a factor of $d$ vertically. The total space has been expanded by a factor $ad$

Everything in the space has grown by a factor $ad$

This is the determinant of the transformation matrix. _the determinant is how much we grow/shrink space_. More precisely, it is the factor by which a given area increases or decreases.

![]({{ site.baseurl }}/images/linear_algebra/determinants1.png){:width="400"}

A more concrete example,

![]({{ site.baseurl }}/images/linear_algebra/determinants5.png){:width="400"}

However, these transformations are not always equal.

Adding $b$ into the matrix;

![]({{ site.baseurl }}/images/linear_algebra/determinants2.png){:width="400"}

The area, the determinant, is still the same $ad$

![]({{ site.baseurl }}/images/linear_algebra/determinants6.png){:width="400"}

To get the area of a general matrix, such as;

$$ A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

The maths gets a little complex; but is $ad - bc$

![]({{ site.baseurl }}/images/linear_algebra/determinants3.png){:width="450"}

This is finding the determinant of A

$\lVert a \rVert = ad - bc$

If a 2D matirx transforms the space such that all points fit onto a single line or even a single point, then the determinant will be 0.

![]({{ site.baseurl }}/images/linear_algebra/determinants7.png){:width="450"}

If a transformation has a determinant of 0, it is transforming the space into a smaller dimension. For a 2-dimenisonal space this is putting everything onto a single line, but the concept applies to $n$ dimensions.

When we "lose a dimensions", the columns must be linearly dependent.

__Orientation__

You can scale an area by a negative amount. As we've seen with vector operations, this does not mean shrinking the area but rather reversing it's orientation. "flipping space". They "invert the orientation of space.

![]({{ site.baseurl }}/images/linear_algebra/determinants8.png){:width="450"}

In a 3D space, the determinant tells us how much volume gets scaled.

### The determinant and inverse matrices

To get the inverse of a 2x2 matrix we flip the terms on the leading diagonal and take the minus of the off-diagonal terms

$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

Multiply these out to get the determinant;

$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \begin{pmatrix} ad-bc & 0 \\ 0 & ad-bc \end{pmatrix}$$

Then multiplying by $\frac{1}{ad-bc}$ gives us the identity matrix

$$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

> The determinant here is what we need to divide the inverse matrix by in order for it to probably be an inverse.

Knowing how to find the determinant in a general sense doesn't add much to the learning process. Unlike the row echelon process we followed previously. Follow `QR decomposition` if interested.

__Linear independence__

Transformations can also remove linear independence.

$A = \begin{pmatrix} 1 & 2 \\ 1 & 2 \end{pmatrix}$ will transform our identity matrix to a single line

$e'_1$ and $e'_2$ become multiples of one another.

The matrix is transforming every point in space along a line. Therefore the determinant is going to be 0.

![]({{ site.baseurl }}/images/linear_algebra/determinants4.png){:width="450"}

If we're working in 3D space, a 3x3 matrix and one of the new basis vectors is just a linear multiple of the other two (ie not linearly independent), the new space is a plane.

> a plane is a flat, two-dimensional surface that extends infinitely far. A plane is the two-dimensional analogue of a point (zero dimensions), a line (one dimension)

Again, as turning a 2D shape into a single dimensional line, the determinant, would be zero. 

For example,

In our matirx
- row 3 = row1 + row2
- column 3 = 2*column1 + column2

$$\begin{pmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 2 & 3 & 7 \end{pmatrix} \begin{bmatrix} a\\b\\c\end{bmatrix} = \begin{pmatrix} 12\\17\\29\end{pmatrix}$$

This matrix doesn't describe three independent basis vectors. It doesn't describe any 3D space but collapses into a 2D space.

As a result, we cannot reduce this to row echelon form.

If you take away the first row from the second, and then the first and second row from the third row, you end up with;

$$\begin{pmatrix} 1 & 1 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix} \begin{bmatrix} a\\b\\c\end{bmatrix} = \begin{pmatrix} 12\\5\\0\end{pmatrix}$$

$0c = 0$ isn't very helpful

There are essentially an infinite number of solutions for $c$, any value of $c$ would work.

As a result, we cannot solve this.

If you think of this from a simultaneous equation point of view, in trying to solve "apples, bananas, and carrots". It's equivalent to the third time going to the copy and ordering a copy of the first two orders. You gathered no new information.

> where the basis vectors describing the matrix are not linearly independent, then the determinant is zero, and that means I can't solve the system of simultaneous equations. Which means I can't invert the matrix because I can't take one over the determinant either. That means I'm stuck, this matrix has no inverse

Doing a transformation that collapses the number of dimensions in space comes at a cost.

The inverse of a matrix, allows us to un-do our transformation. You cannot un-do, moving from a 3D space to a 2D plane.

### Python and Matrices

Write a function that will test if a 4×4 matrix is singular, i.e. to determine if an inverse exists, before calculating it.

You shall use the method of converting a matrix to echelon form, and testing if this fails by leaving zeros that can’t be removed on the leading diagonal.

```python
import numpy as np

def isSingular(A) :
    B = np.array(A, dtype=np.float_) 
    """Make B as a copy of A. We're going to alter it's values"""
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False

"defines our error flag"
class MatrixIsSingular(Exception): pass    
```

```python
"""
For Row Zero, all we require is the first element is equal to 1.
We'll divide the row by the value of A[0, 0].
This will get us in trouble though if A[0, 0] equals 0
    , so first we'll test for that,
If this is true, we'll add one of the lower rows to the first one before the division.
We'll repeat the test going down each lower row until we can do the division.
There is no need to edit this function.
"""
def fixRowZero(A) :
    if A[0,0] == 0 :
        A[0] = A[0] + A[1]
    if A[0,0] == 0 :
        A[0] = A[0] + A[2]
    if A[0,0] == 0 :
        A[0] = A[0] + A[3]
    if A[0,0] == 0 :
        raise MatrixIsSingular()
    A[0] = A[0] / A[0,0]
    return A
```

```python
"""First we'll set the sub-diagonal elements to zero, i.e. A[1,0].
Next we want the diagonal element to be equal to one.
We'll divide the row by the value of A[1, 1].
Again, we need to test if this is zero.
If so, we'll add a lower row and repeat setting the sub-diagonal elements to zero.
There is no need to edit this function."""
def fixRowOne(A) :
    A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1,0] * A[0]
    if A[1,1] == 0 :
        raise MatrixIsSingular()
    A[1] = A[1] / A[1,1]
    return A
```

```python
def fixRowTwo(A) :
    """Insert code below to set the sub-diagonal elements of row two to zero (there are two of them)."""
    A[2] = A[2] - A[2,0] * A[0]
    A[2] = A[2] - A[2,1] * A[1]
    
    "Next we'll test that the diagonal element is not zero."
    if A[2,2] == 0 :
        "Insert code below that adds a lower row to row 2."
        A[2] = A[2] + A[3]
        "Now repeat your code which sets the sub-diagonal elements to zero."
        A[2] = A[2] - A[2,0] * A[0]
        A[2] = A[2] - A[2,1] * A[1]
        
    if A[2,2] == 0 :
        raise MatrixIsSingular()
    "Finally set the diagonal element to one by dividing the whole row by that element."
    A[2] = A[2] / A[2,2]
    return A

def fixRowThree(A) :
    """Insert code below to set the sub-diagonal elements of row three to zero."""
    A[3] = A[3] - A[3,0] * A[0]
    A[3] = A[3] - A[3,1] * A[1]
    A[3] = A[3] - A[3,2] * A[2]
    
    "Complete the if statement to test if the diagonal element is zero."
    if A[3,3] == 0:
        raise MatrixIsSingular()
    "Transform the row to set the diagonal element to one."
    A[3] = A[3] / A[3,3]
    return A
```

## Matrices make linear mappings

### Einstein summation convention and the symmetry of the dot product

The Einstein summation convention is a way to write matrix transformations. It writes down what the actual operations are on the elements of the matrix.

When we started, we said that multiplying a matrix by a vector or with another matrix is a process of taking every element in each row in turn, multiplied with corresponding element in each column in the other matrix, and adding them all up and putting them in place.

![]({{ site.baseurl }}/images/linear_algebra/einsteinconvention1.png){:width="400"}

Einstein convention,says, you have a sum over some elements $j$, for all the possible combinations of $i$ and $k$. As this is a repeated index, don't bother with the sum;

$$ ab_{ik} = \sum_{j} a_{ij} b_{jk} = a_{ij} b_{jk}$$

We'd only have to run `for loops` over $i$, $j$ and $k$. Then use an accumulator on the $j$'s to find the elements of the product matrix $AB$.

We can multiply matrices that are not square (ie same numbers of rows in A as the same number of columns in B)

As long as you have the same number of $j$s you can multiply them together.

![]({{ site.baseurl }}/images/linear_algebra/einsteinconvention2.png){:width="400"}

> Now, all sorts of matrix properties that you might want, inverses and so on, determinants, all start to get messy and mucky, and you sometimes can't even compute them when you're doing this sort of thing.

__Revisiting the dot product__

If we have two column vectors (a single column with n elements);

$U = \begin{bmatrix} u_i \end{bmatrix}$ and $V = \begin{bmatrix} v_i \end{bmatrix}$

The summation convention is $u_i v_i$, we repeat over all the $i$s and add.

That's the same as doing a matrix transformation, where $u$ is a row and $v$ a column.

![]({{ site.baseurl }}/images/linear_algebra/einsteinconvention3.png){:width="400"}

Projection is symmetric. Projection is the dot product.

If you project $\hat{u}$ down onto $\hat{e_1}$ and vice versa, the two triangles split by the pink line are identical.

![]({{ site.baseurl }}/images/linear_algebra/einsteinconvention4.png){:width="400"}

> there is this connection between this numerical thing, matrix multiplication, and this geometric thing, projection

That's why we talk about a matrix multiplication with a vector as being the projection of that vector onto the vectors composing the matrix, the columns of the matrix

### Non-square matrix multiplication

In traditional notion we might write $\sum^3 A_{ij} v_j = A_{i1} v_1 + A_{i2} v_2 + A_{i3} v_3$

With Einstein summation convention we can simply write $A_{ij}v{j}$ and know that we sum over $j$ because it appears twice.

We can multiply any matrices together as long as the terms which we sum over have the same number of elements
- need to clarify

We can multiply an $m \times n$ matrix with an $n \times k$ matrix, and the resultant matrix will be an $m \times k$ matrix.

$$A=\begin{bmatrix} 1 & 2 & 3 \\ 4 & 0 & 1\end{bmatrix} B=\begin{bmatrix} 1 & 1 & 0\\0 & 1 & 1\\1 & 0 & 1\end{bmatrix}$$

$$C_{mn} = A_{mj} B{jn}$$

$$C_{21} = A_{2j} B_{j1} = 5$$

Equivalent to $C_{21} = A_{2j} B_{j1} = A_{21} B_{11} + A_{22} B_{21} + A_{23} B_{31}$

```python
(4*1) + (0*0) + (1*1) = 5
```

elements in second in A with elements in first column in B

$C_{11}$ for example is;
```python
(1*1) + (2*0) + (3*1) = 4
```

$$C=\begin{bmatrix} 4 & 3 & 5 \\ 5 & 4 & 1\end{bmatrix}$$

__Calculate the product__

`Question`

$$\begin{bmatrix} 2 & 4 & 5 & 6\end{bmatrix} \begin{bmatrix} 1 \\ 3 \\ 2 \\ 1\end{bmatrix}$$

`Answer 30`

`Question`

$$\begin{bmatrix} 1 \\ 3 \\ 2 \\ 1\end{bmatrix} \begin{bmatrix} 2 & 4 & 5 & 6\end{bmatrix}$$

`Answer, produces a 4x4`

`Question`

$$\begin{bmatrix} 2 & 4 & 5 & 6 \\6 & 12 & 15 & 18\\4 & 8 & 10 & 12\\2 & 4 & 5 & 6\end{bmatrix}$$

```
- the row is repeated each time
- multiplied by the relevant element in the first vector
```
`Question`

$$\begin{bmatrix} 2 & -1 \\ 0 & 3 \\ 1 & 0\end{bmatrix} \begin{bmatrix} 0 & 1 & 4 & -1 \\ -2 & 0 & 0 & 2\end{bmatrix}$$

```python
# Answer
# Column1;
(2*0) + (-1*-2) = 2
(0*0) + (3*-2) = -6
(1*0) + (0*-2) = 0

# Column2;
(2*1) + (-1*0) = 2
(0*1) + (3*0) = 0
(1*1) + (0*0) = 1

# Column3;
(2*4) + (-1*0) = 8
(0*4) + (3*0) = 0
(1*4) + (0*0) = 4

# Column3;
(2*-1) + (-1*2) = -4
(0*-1) + (3*2) = 6
(1*-1) + (0*2) = -1

```

$$\begin{bmatrix} 2 & 2 & 8 & -4 \\ -6 & 0 & 0 & 6 \\ 0 & 1 & 4 & -1\end{bmatrix}$$

`Question`

$D = ABC$ where 
- A is a 5*3 matrix
- B is a 3*7 matrix
- C is a 7*4 matrix

What are the dimensions of $D$?

The size of $AB$ is 5*7

The size of $(AB)C$ == $A(BC)$ therefore the matrices can be multiplied together.

$D$ is a 5*4 matrix

`Question`

Calculate the product;

$$\begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}$$

The identity matrix doesn't change the second matrix, so remains $\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}$

Let $\mathbf{u}$ and $\mathbf{v}$ be vectors with $n$ elements. Write the dot product;

--TODO

### Using non-square matrices to do a projection

Shadows are an example of a transformation that reduces the number of dimensions. For example, 3D objects in the world cast shadows on surfaces that are 2D.

![]({{ site.baseurl }}/images/linear_algebra/nonsquareprojections1.png){:width="400"}

The sun is sufficiently far away that effectively all of its rays come in parallel to each other. We can describe their direction with the unit vector $\hat{\mathbf{s}}$

We can describe the 3D coordinates of points on objects in our space with the vector $\mathbf{r}$. Objects will cast a shadow on the ground at the point $\mathbf{r}'$  along the path that light would have taken if it hadn't been blocked at $\mathbf{r}$, that is, $\mathbf{r}' = \mathbf{r} + \lambda \hat{\mathbf{s}}$

The ground is at $\mathbf{r}'_3 = 0$; by using $\mathbf{r}'.\hat{\mathbf{e}}_3 = 0$, we can derive the expression, $\mathbf{r}.\hat{\mathbf{e}}_3 + \lambda s_3 = 0$, (where $s_3 = \hat{\mathbf{s}}.\hat{\mathbf{e}}_3$).

Rearrange this expression for $\lambda$ and substitute it back into the expression for $\mathbf{r}'$, in order to get $\mathbf{r}'$ in terms of $\mathbf{r}$.

$$ \mathbf{r}' = r - \hat{s}(r.\hat{e_3})/s_3$$

Remember that;
The scalar projection of $s$ on $r$;

$$ proj_rs = \frac{s.r}{\lVert r \rVert}$$

$\mathbf{r}'$ can be written as a linear transformation of $r$. This means we should be able to write $\mathbf{r}' = Ar$ for some matrix $A$

Usiing Einstein summation convention, we can rewrite our previous answer as;
$$r'_i = r_i - s_i[\hat{e_3}]_jr_j/s_3$$
or
$$r'_i = r_i - s_ir_3/s_3$$
or
$$r'_i = (I_{ij} - s_i[\hat{e_3}]_j/s_3)r_j$$
or
$$r'_i = (I_{ij} - s_iI_{3j}/s_3)r_j$$

We can now give an expression for $A$ in its component form by evaluating the components A_{ij} for each row $i$ and column $j$.

$A$ takes a 3D vector $r$ and transforms it into a 2D vector $r'$. Therefore the matrix will be a 2x3.
    - remember; the columns of a matrix are the vectors in the new space that the unit vectors of the old space transform to

$$ A = \begin{bmatrix} 1 & 0 & -s_1/s_3\\ 0 & 1 & -s_2/s_3\end{bmatrix}$$

If you were to evaluate it's third row it would be all zeros as $r'$ has no value in the third dimension;
$A3 = [0, 0, 0]$

Assume the Sun's rays come in the direction;
$$ \hat{s} = \begin{bmatrix} 4/13 \\ -3/13 \\ -12/13 \end{bmatrix}$$

Construct the matrix A, apply it to a point, on an object in our space to find the coordinate of that point's shadow;

$r = \begin{bmatrix} 6 \\ 2 \\ 3 \end{bmatrix}$, $A = \begin{bmatrix} 1 & 0 & -4/13 / -12/13\\ 0 & 1 & 3/13 / -12/13\end{bmatrix}$

$r' = A_{ij}r{j}$

```python
# r'
(1 * 6) + (0 * 2) + (0.333 * 3)
(0 * 6) + (1 * 2) + (-0.25 * 3)

[7, 1.25]
```

Another use of non-square matrices is applying a matrix to a list of vectors.

![]({{ site.baseurl }}/images/linear_algebra/nonsquare1.png){:width="600"}

Observe that it's the same result as treating the columns as separate vectors and calculating them individually.

Using $\hat{s} = \begin{bmatrix} 4/13 \\ -3/13 \\ -12/13 \end{bmatrix}$, apply A to the matrix;

$R = \begin{bmatrix} 5&-1&-3&-7 \\ 4&-4&1&-2 \\ 9&3&0&12\end{bmatrix}$

$A = \begin{bmatrix} 1 & 0 & 0.333\\ 0 & 1 & 0.25\end{bmatrix}$

```python
# R`
Rp = [[a,  b,  c,  d],
      [e,  f,  g,  h]]

a = (1 * 5) + (0 * 4) + ((-4/13)/(-12/13) * 9)
e = (0 * 5) + (1 * 4) + (-0.25 * 9)

b = (1 * -1) + (0 * -4) + ((-4/13)/(-12/13) * -3)
f = (0 * -1) + (1 * -4) + (-0.25 * -3)

c = (1 * -3) + (0 * 1) + ((-4/13)/(-12/13) * 0)
g = (0 * -3) + (1 * 1) + (-0.25 * 0)

d = (1 * 7) + (0 * -2) + ((-4/13)/(-12/13) * 12)
h = (0 * 7) + (1 * -2) + (-0.25 * 12)

Rp = [[8.0, -2.0, -3.0, 11.0],
      [1.75, -3.25, 1.0, -5.0]]
```

## Matrices transform into the new basis vector set

The columns of a transformation matrix, are the axes of the new basis vectors of the mapping in my coordinate system.

The lines in yellow describe the world of Panda Bear. To him, these vectors are [1,0] and [0,1] but in my frame, they are [3,1] and [1,1]

![]({{ site.baseurl }}/images/linear_algebra/matrix_basis1.png){:width="500"}

Bear's transformation matrix is therefore; $\begin{bmatrix} 3&1 \\ 1&1 \end{bmatrix}$

Now if we take a vector in Bear's world, we can understand it in my coordinate system.

The transformation matrix is Bear's basis vectors in my coordinate system.

The vector `1/2[3, 1]` or `[3/2, 1/2]` in Bear's world becomes;

`[(3 * 3/2) + (1 * 1/2), (1 * 3/2) + (1 * 1/2)]` or `[5, 2]`

![]({{ site.baseurl }}/images/linear_algebra/matrix_basis2.png){:width="500"}

However, we need to figure out how to go the other way. Translating my world to Bear's world.

To get my basis vectors in Bear's coordinates, we need to take the inverse of the transformation matrix;
- flip on the leading diagonal, and put a minus on the off diagonal terms
- Recall that when a matrix is transformed into its diagonal form, the entries along the diagonal are the eigenvalues of the matrix - this can save lots of calculation!

- $\begin{bmatrix} 1&-1 \\ -1&3 \end{bmatrix}$

- then divide by the determinant (three minus one over one, 3-1/1 = 2), so multiply by a half 

- $\frac{1}{2} \begin{bmatrix} 1&-1 \\ -1&3 \end{bmatrix}$

- remember - _the determinant is how much we grow/shrink space_

__Example__

Here Bear's world is going to be an orthonormal basis vector set (they form a v)

_you can to a dot product to verfiy they are at 90 degrees to one another (orthogonal)_

![]({{ site.baseurl }}/images/linear_algebra/matrix_basis3.png){:width="400"}

Bear's transformation matrix, ie the matrix that converts a vector in Bear's world to my coordinate system;

$B = \frac{1}{\sqrt2} \begin{bmatrix} 1&-1 \\ 1&1 \end{bmatrix}$

The inverse, or $B^{-1}$

$B = \frac{1}{\sqrt2} \begin{bmatrix} 1&1 \\ -1&1 \end{bmatrix}$

Because Bear's vectors are orthogonal, we can do this using projections, rather than having to calculate the transformation matrices.

Take my version of the vector and dot it with Bear's axis, then we get the answer of the vector in Bear's world;

first component, the vector * the first axis;
$$\frac{1}{\sqrt2} \begin{bmatrix} 1 \\ 3 \end{bmatrix} . \frac{1}{\sqrt2} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{2} 4 = 2$$

```python
1/sqrt{2} * 1/sqrt{2} = 1/2
(3 * 1) + (1 * 1) = 4
```

second component, the vector * the second axis;
$$\frac{1}{\sqrt2} \begin{bmatrix} 1 \\ 3 \end{bmatrix} . \frac{1}{\sqrt2} \begin{bmatrix} -1 \\ 1 \end{bmatrix} = 1$$

```python
1/sqrt{2} * 1/sqrt{2} = 1/2
(3 * 1) + (1 * -1) = 2
```

In this case the lengths are one. When using projections normally we would need to normalise by their lengths (see scalar projections).

If Bear's vectors are orthogonal to one another, we don't have to use the matrix transformations, we can use the dot product.

### Doing a transformation in a changed basis

Doing a 45 degree rotation in my world is done using the following matrix;

$$\frac{1}{\sqrt2}  \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation_newbasis1.png){:width="400"}

However, we don't know what a 45 degree transformation looks like in Bear's world.

First, we need to transform the vector into our world using the matrix $B$, which is the Bear's basis vector in our world;

$B = \begin{bmatrix} 3&1 \\ 1&1 \end{bmatrix}$

Then apply the transformation,

Then reverse the result using $B^{-1}$

$B^{-1} = \frac{1}{2} \begin{bmatrix} 1&-1 \\ -1&3 \end{bmatrix}$

Or written as; $B^{-1} R B = R_B$

![]({{ site.baseurl }}/images/linear_algebra/matrix_transformation_newbasis2.png){:width="400"}

> If we want to transform to normal form normal coordinate systems, then the translation matrices also change. We have to be mindful of that
>
> We've got the transformation matrix R, wrapped around by B, B is the minus 1, that does the translation from my world to the world of the new basis system.

__Practice__

Finish off the calculation;

$$\frac{1}{2} \begin{bmatrix} 3&-1 \\ -1&1\end{bmatrix} * \frac{1}{\sqrt2} \begin{bmatrix} 2&0 \\ 4&2\end{bmatrix} = \frac{1}{\sqrt2} \begin{bmatrix} 1&-1 \\ 1&1\end{bmatrix}$$

```python
# working
# first matrix row * second matrix column
x1y1 = (1.5 * 2) + (-0.5 * 4) = 1
x1y2 = (1.5 * 0) + (-0.5 * 2) = -1
x2y1 = (-0.5 * 2) + (0.5 * 4) = 1
x2y2 = (-0.5 * 0) + (0.5 * 2) = 1
```

### Orthogonal matrices

We can transpose a matrix, where we interchange all the elements of the rows and columns of the matrix; $A^T_{ij} = A_{ij}$

$$\begin{bmatrix} 1&2 \\ 3&4\end{bmatrix}^T = \begin{bmatrix} 1&3 \\ 2&4\end{bmatrix}$$

We interchange the elements that are off the diagonal. So the one and the four stay where they are. 

Because if is, I find $i$ and $j$ are the same; elements at the coorindates 1,1 would stay the same, the same for 2,2. But the element 1,2 interchange to the element 2,1.

Imagine we have a square matrix $A$, with dimensions n x n. This defines a transformation (ie we apply to vectors to transform them), the columns in this matrix are the basis vectors in a new space.

In this matrix;

- the vectors are orthogonal to each other and of unit length 
    - $a_i . a_j = 0$ if $i \neq j$
    - $a_i . a_j = 1$ if $i = j$

If we multiply this matrix by it's transpose $A^T$, the columns $a_1$ to $a_n$ become rows.

The result of the multiplication is an identity matrix;

![]({{ site.baseurl }}/images/linear_algebra/orthogonal_matrices1.png){:width="400"}

$A^T$ is a valid inverse of $A$

> A set of unit length basis vectors that are all perpendicular to each other are called an __orthonormal basis set__

They must meet the criteria;
- $a_i . a_j = 0$ if $i \neq j$
- $a_i . a_j = 1$ if $i = j$

> The matrix composed of them is called an orthogonal matrix

Because all the basis vectors in an orthogonal matrix are of unit length, it must scale space by a factor of one.

The determinant of an orthogonal matrix must be either plus or minus one (depending on if you do $A^T * A or A * A^T$).

We want our transformation matrix to be an orthogonal matrix, we want the basis vectors to be orthonormal.

- the inverse is easy to compute
- the transformation is reversible as it doesn't collapse the space
- the projection is the dot product
- if arranged in the right order the determinant is one

### The Gram-Schmidt process

If we assume we already have some linearly independent vectors that span the space we're interested in, we can construct an orthonormal basis vector set.

We can check linear independence by calculating the determinant. If there are linearly independent the determinant will be 0.

![]({{ site.baseurl }}/images/linear_algebra/grahmschmidt1.png){:width="400"}

However, these vectors are not orthogonal to one another and are not of unit length.

The Gram-Schmidt process allows us to transform this vector set into an orthonormal set.

Take the first vector $v_1$ and normalise it to be on unit length; 

$e_1 = \frac{v_1}{\lVert v_1 \rVert}$

$v_2$ can now be thought of as (1) a component of that's in the direction of $e_1$ plus (2) a component that is perpendicular to $e_1$, which we can find by taking the projection of $v_2$ onto $e_1$

![]({{ site.baseurl }}/images/linear_algebra/grahmschmidt2.png){:width="250"}

The vector projection is $\frac{(v_2 . e_1)}{\lVert e_1 \rVert}$ or simply ($v_2 . e_1$) as $e_1$ has already been normalised, it has size 1. To get this as a vector, not just a number we multiply by $e_1$

$v_2 = (v_2 . e_1) e_1$ + $u_2$

Rewritten as;

$u_2 = v_2 - (v_2 . e_1) e_1$

Then normalised to unit length

$e_2 = \frac{u_2}{\lVert u_2 \rVert}$

![]({{ site.baseurl }}/images/linear_algebra/grahmschmidt3.png){:width="400"}

$v_3$ is not linear combination of $v_1$ and $v_2$, therefore it is not in the plane defined by $v_1$ and $v_2$ and following that, will not be defined by $e_1$ and $e_2$.

As a result, we need to project $v_3$ down onto the plane of $e_1$ and $e_2$ and the projection will be some vector in the plane composed of $e_1$s and $e_2$s

The perpendicular vector $u_3$ is what is left when you remove the elements of $v_3$ made up of $e_1$ and $e_2$

$u_3 = v_3 - (v_3 . e_1)e_1 - (v_3 . e_2)e_2$

Then normalise;

$e_3 = \frac{u_3}{\lVert u_3 \rVert}$

$e_1, e_2, e_3$ now form an orthonormal basis set

We can write a function to perform the Gram-Schmidt procedure. Taking in a list of vectors and forming an orthonormal basis set.

```python
# Remember we access elements in matrix as;
# individual elements
A[n, m]
# rows
A[n]
# columns
A[:, m]

# calculate the dot product using @
u @ v
```

Say we have 4 basis vectors;

```python
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# Our first function will perform the Gram-Schmidt procedure for 4 basis vectors.
# We'll take this list of vectors as the columns of a matrix, A.
# We'll then go through the vectors one at a time and set them to be orthogonal
# to all the vectors that came before it. Before normalising.
def gsBasis4(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # The zeroth column is easy, since it has no other vectors to make it normal to.
    # All that needs to be done is to normalise it. I.e. divide by its modulus, or norm.
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    # For the first column, we need to subtract any overlap with our new zeroth vector.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # If there's anything left after that subtraction, then B[:, 1] is linearly independant of B[:, 0]
    # If this is the case, we can normalise it. Otherwise we'll set that vector to zero.
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])
    # Now we need to repeat the process for column 2
    # Subtract the overlap with the zeroth vector,
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    # Subtract the overlap with the first.
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]
    # Again we'll need to normalise our new vector.
    if la.norm(B[:, 2]) > verySmallNumber :
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else :
        B[:, 2] = np.zeros_like(B[:, 2])    
    # Finally, column three:
    # Subtract the overlap with the first three vectors.
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]    
    
    # Now normalise if possible
    if la.norm(B[:, 3]) > verySmallNumber :
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else :
        B[:, 3] = np.zeros_like(B[:, 3])     
    
    # Finally, we return the result:
    return B
```

However, we can generalise the procedure;

```python
def gsBasis(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # Loop over all vectors, starting with zero, label them with i
    for i in range(B.shape[1]) :
        # Inside that loop, loop over all previous vectors, j, to subtract.
        for j in range(i) :
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        # do the normalisation test for B[:, i]
        if la.norm(B[:, i]) > verySmallNumber :
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else :
            B[:, i] = np.zeros_like(B[:, i])
            
    # Finally, we return the result:
    return B

# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))
```

### Reflecting in a plane

Say we want to know what a vector looks like when reflected in some plane.

We have three vectors, the first two are within the plane of the mirror, the third is outside the plane.

$$v_1 = \begin{bmatrix} 1\\1\\1\end{bmatrix} v_2 = \begin{bmatrix} 2\\0\\1\end{bmatrix} v_3 = \begin{bmatrix} 3\\1\\-1\end{bmatrix}$$

![]({{ site.baseurl }}/images/linear_algebra/reflecting.png){:width="200"}

Using the Grahm-Schimdt procedure to get some orthonormal basis vectors that desrribe the plane and its normal $v_3$

$e_1$ is the normalised version of $v_1$. $v_1$ is a length 3 ($\sqrt{1^2 + 1^2 + 1^2}$)

$$ e_1 = \frac{v_1}{\lVert v_1 \rVert} = \frac{1}{\sqrt3} \begin{bmatrix} 1\\1\\1\end{bmatrix}$$

$u_2$ is $v_2$ minus some number of $e_1$s -> more precisely; the projection of $v_2$ onto $e_1$ ($v_2 . e_1$) multiplied by $e_1$

$$u_2 = v_2 - (v_2 . e_1)e_1 = \begin{bmatrix} 2\\0\\1\end{bmatrix} - (\begin{bmatrix} 2\\0\\1\end{bmatrix} . \frac{1}{\sqrt3} \begin{bmatrix} 1\\1\\1\end{bmatrix}) \frac{1}{\sqrt3} \begin{bmatrix} 1\\1\\1\end{bmatrix}$$

```python
the root threes come outside, so become 1/3 
[2, 0, 1] dotted with [1, 1, 1] is 3
so they cancel out to 1

so [2, 0, 1] - [1, 1, 1]
```

$$ = \begin{bmatrix} 1\\-1\\0\end{bmatrix}$$

$e_2$ is equal to the normalised version of $u_2$

$$e_2 = \frac{u_2}{\lVert u_2 \rVert} = \frac{1}{\sqrt2} \begin{bmatrix} 1\\-1\\0\end{bmatrix}$$

Then we need $u_3$

$$u_3 = v_3 - (v_3 . e_1)e_1 - (v_3 . e_2)e_2 = \begin{bmatrix} 1\\1\\-2\end{bmatrix}$$

$$e_3 = \frac{1}{\sqrt6} \begin{bmatrix} 1\\1\\-2\end{bmatrix}$$

Our new transformation is $E$, described by our new basis vectors

$$E = \begin{pmatrix} \begin{bmatrix}e_1\end{bmatrix} \begin{bmatrix}e_2\end{bmatrix} \begin{bmatrix}e_3\end{bmatrix} \end{pmatrix}$$

This contains the plane ($e_1$ and $e_2$ and then the normal to the plane $e_3$. _It's the bit of v3 that we can't make by projecting on to v1 and v2, then of unit length_)

Say we have some vector $r$ that we want to reflect down through the pane, and get $r'$ on the other side;

![]({{ site.baseurl }}/images/linear_algebra/reflecting2.png){:width="400"}

We can think of $r$ as composed of some vector within the pane (composed of $e_1$s and $e_2$s) - this the dotted line perpendicular to $e_1$ and $e_2$ and some vector that is normal (ie made up of $e_3$s) - this is the dotted line up to $r$

When we reflect through the pane, the bit made up of $e_1$ and $e_2$ will be unchanged and the bit composed of $e_3$s will be inverted.

$$T_E = \begin{pmatrix} \begin{bmatrix}e_1\end{bmatrix} \begin{bmatrix}e_2\end{bmatrix} \begin{bmatrix}e_3'\end{bmatrix} \end{pmatrix}$$

Getting from $r$ to $r'$ is hard. As we saw with Bear. 

- First we need to transform $r$ into the basis plane using the inverse of our orthogonal basis matrix,
- Then transform it, do the reflection in the basis of the plane
- Then read that back into my basis using the orthogonal basis matrix.

_Note_ in this example we're changing from our basis vectors to the plane's and then translating back into our basis. So the $E$ and the $E^{-1}$ are flipped compared to when we were working with Bear.

![]({{ site.baseurl }}/images/linear_algebra/reflecting3.png){:width="250"}

$E T_E E^{-1}r = r'$

Written out the whole thing looks a little ungodly, but that's more down to the volume of arthimetic required than the complexity

![]({{ site.baseurl }}/images/linear_algebra/reflecting4.png){:width="400"}

Generalised, this is the process of reflecting a point in space in a mirror (we can transform something through the looking glass). 

In the _practical_ world of machine learning, this will be the technique used when transforming images of forces for the purpose of doing facial recognition. You transform a face from being side on to profile, and then using some form of neural network to do the recognition.

__Practice reflections__

Perform a transformation that is easy in a particular basis, but complicated in our starting basis.

Namely we shall help Panda Bear determine what his reflection will look like in a mirror that he has placed at an angle.

The mirror lies along the first axis. But, as is the way with bears, his coordinate system is not orthonormal: so what he thinks is the direction perpendicular to the mirror isn't actually the direction the mirror reflects in

Write a Python function that will produce a transformation matrix for reflecting vectors in an arbitrarily angled mirror.

$T = E T_E E^{-1}$

_note_
> the @ operator is used to combine vectors and/or matrices in the expected linear algebra way,
i.e. it will be either the vector dot product, matrix multiplication, or matrix operation on a vector, depending on it's input.

> This is in contrast to the \\(*\\) operator, which performs element-wise multiplication, or multiplication by a scalar.

```python
import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose
from readonly.bearNecessities import *

def build_reflection_matrix(bearBasis):
"""Return the transformation T"""
# built using an orthonormal basis set (E), created from Bear's Basis
# and a transformation matrix (TE) in the mirror ccoordinates"""

    # Use the gsBasis function on bearBasis to get the mirror's orthonormal basis
    E = gsBasis(bearBasis) # bearBasis is a 2×2 matrix

    # Write a matrix in component form that performs the mirror's reflection in the mirror's basis
    # the mirror operates by negating the last component of a vector - one axis doesn't change
    TE = np.array([0, 1],
                   [0, -1]])

    # Combine the matrices E and TE to produce your transformation matrix.
    T = E @ TE @ inv(E) 

    return T
```

## Eigenvalues and Eigenvectors

We'll start by using geometric expressions (shapes in 2d) to conceptually understand "_eigen-ness_"

Eigen is perhaps most usefully translated from German to mean charactersitic. When we talk about an eigenproblem, we're talking about finding the charactersitic properities of something.

We've seen that we can express linear transformations using matrices. These transformation operations include scalings, rotations and shears.

> A transformation in which all points along a given line L remain fixed while other points are shifted parallel to L by a distance proportional to their perpendicular distance from L. Shearing a plane figure does not change its area. The shear can also be generalized to three dimensions, in which planes are translated instead of lines. - Wolfram

![]({{ site.baseurl }}/images/linear_algebra/shear1.gif){:width="250"}

Typically, we have thought about how these transformations change a single vector. What about if the matrix was applied to all vectors in the space?

We can think of this by having a square, centred in the middle of our basis vectors and seeing how the shape is transformed.

Applying a scaling of 2 in the vertical direction, the square becomes a rectangle. Applying a horizontal sheer gives us a parrallelegram.

![]({{ site.baseurl }}/images/linear_algebra/eigen1.png){:width="400"}

The square helps us to understand what is happening to many vectors. However, some vectors remain on the same line they started on, while others do not.

Take our initial square, with three vectors drawn on;

If we scale vertically, the diagonal vector will __not__ be pointing in the same direction.
Any other vector's direction would have changed (apart from the horizontal and vertical, their angle and size will have changed)

![]({{ site.baseurl }}/images/linear_algebra/eigen2.png){:width="250"}

The horizontal and vertical vectors are charactersitic of this particular trnasformation. They are the only ones that do not change. They are referred to as eigenvectors.

Becuase the horiztonal's length was unchanged, we say it has a "corresponding eigenvalue of one", whereas the vertical eigenvector doubled in length, so it has a "corresponding eigenvalue of two".

Eigenvectors are those laying on the same span as before the transformation. Then we measure how much their length has changed.

Take a pure shear (where there is no rotation or scaling so the area is unchanged), here we would have one eigenvector, along the horiztonal;

![]({{ site.baseurl }}/images/linear_algebra/eigen3.png){:width="250"}

In rotation, there are no eigenvectors.

![]({{ site.baseurl }}/images/linear_algebra/eigen4.png){:width="250"}

__Practice__

In all examples, we'll start with the following vectors and apply a transformation $T$ 

![]({{ site.baseurl }}/images/linear_algebra/eigen5.png){:width="250"}

$T_1= \begin{bmatrix} 2&0\\0&2 \end{bmatrix}$

$T_1$ scales each vectors by 2, so all three can be considered eigenvectors.

$T_2= \begin{bmatrix} 3&0\\0&2 \end{bmatrix}$

$T_2$ scales the x axis by 3 and the y axis by 2, so our purple vector will no long be on the same plane.

$T_3= \begin{bmatrix} 1&2\\0&1 \end{bmatrix}$

$T_3$ is a sheer, the x axis is unchanged, but the angle and size of the other two vectors is changed along the x-axis

$T_4= \begin{bmatrix} 0&-1\\1&0 \end{bmatrix}$

$T_4$ is an anti-clockwise rotation, so there will be no vectors that remain pointing along the same pane.

$T_4= \begin{bmatrix} 0&-1\\1&0 \end{bmatrix}$

$T_4$ is an anti-clockwise rotation, so there will be no eigenvectors

$T_5= \begin{bmatrix} -1&0\\0&-1 \end{bmatrix}$

$T_5$ is a reflection, all vectors will be pointing along the same pane, just in the opposite direction

$T_6= \begin{bmatrix} 2&1\\0&2 \end{bmatrix}$

$T_6$ scales all vectors by 2, but alters the angle at which the orange and purple vectors are pointing

![]({{ site.baseurl }}/images/linear_algebra/eigen6.png){:width="650"}

To summaise, eigenvectors are those that lay along the same path after applying a linear transformation to a space. Eigenvalues are the amount that each of those vectors has been stretched in the process (or negative if flipped).

In the practice examples, it appeared that rotation would leave us without any eigenvectors, however, 180 degree rotation, is equivalent to a reflection, where the vectors are pointing in the opposite direction. All vectors will be eigenvectors will an eigven value of -1.

A transformation that is some combination of horizontal shearing and vertical scaling does have two eigenvectors.
The first which is most obvious is the horiztonal vector. The second is between the organge and the pink vector. Though the concept is straight forward, there are not always easy to spot.

![]({{ site.baseurl }}/images/linear_algebra/eigen7.png){:width="400"}

This problem is amplified in three or more dimensions, where we can't simply use geometric representations to spot eigenvectors.

In 3D scaling and shearing work much the same way, but rotation works differently. The eigenvector represents the axis of rotation.

![]({{ site.baseurl }}/images/linear_algebra/eigen8.png){:width="400"}

### Eigenvectors, a formal definition

Alegrabically eigenvectors ($x$) can be represented as; 

$Ax = \lambda x$

On the lefthand side $A$ represents a transformation matrix being applied to the vector $x$. On the righthand side we are stretching the vector by some scalar factor lambda.

$A$ must be a square transformation ($n \times n$) and $x$ must be an $n$ dimensional vector. Otherwise it's shape would change, it wouldn't just scale.

$(A - \lambda I) x = 0$ _I represents an identity matrix, the same size as $A$_

> We didn't need this in the first expression we wrote, as multiplying vectors by scalars is defined. However, subtracting scalars from matrices is not defined

Either $x$ is 0 or the contents of the brackets. However, we're not interested we $x = 0$, that means the vector has no length or direction, it's a trivial solution.

We can test if a matrix operation will result in a 0 output by calculating its determinant.

$det(A - \lambda I) = 0$

In the case of a 2x2;

$$det\begin{pmatrix} \begin{pmatrix} a&b\\c&d \end{pmatrix} - \begin{pmatrix} \lambda&0 \\ 0&\lambda \end{pmatrix} \end{pmatrix} = 0$$

Evaluating this determinant, we get what is referred to as the _characteristic polynomial_

$$\lambda^2 - (a+d)\lambda + ad - bc = 0$$

Our eigenvalues are simply the solutions of this equation, and we can then plug these eigenvalues back into the original expression to calculate our eigenvectors.

This gets complex in high dimensions, but that's why we have computers (they truly are a bicycle for the mind).

Let's take the example of a vertical scaling of 2

$A = \begin{pmatrix} 1&0\\0&2 \end{pmatrix}$

Take the determinant of A minus lambda I ($A - \lambda I$) and set it to zero

$$det\begin{pmatrix} 1-\lambda&0\\0&2-\lambda \end{pmatrix} = 0 = (1-\lambda)(2-\lambda)$$

Our equation has solutions at lambda = 1 and lambda = 2, so we can substitute back in;

$$@\lambda=1: \begin{pmatrix} 1-1&0\\0&2-1\end{pmatrix} \begin{bmatrix} x_1\\x_2 \end{bmatrix} = \begin{pmatrix} 0&0\\0&1\end{pmatrix} \begin{bmatrix} x_1\\x_2 \end{bmatrix} = \begin{bmatrix} 0\\x_2\end{bmatrix} = 0$$

$$@\lambda=2: \begin{pmatrix} 1-2&0\\0&2-2\end{pmatrix} \begin{bmatrix} x_1\\x_2 \end{bmatrix} = \begin{pmatrix} -1&0\\0&0\end{pmatrix} \begin{bmatrix} x_1\\x_2 \end{bmatrix} = \begin{bmatrix} -x_1\\0\end{bmatrix} = 0$$

At lambda equals one, the $x_2$ term must = 0. Our x axis $x_1$ can equal anything, as long as there's 0 in the vertical direction.

We express this as; $@\lambda=1: x = \begin{bmatrix} t\\0 \end{bmatrix}$

At lambda equals two, we can express our eigen vector as not moving in the horizontal direction, any scaling along the vertical axis.

We express this as; $@\lambda=2: x = \begin{bmatrix} 0\\t \end{bmatrix}$

Let's take the example of a 90 degree rotation, where we expect no eigenvectors.

$A = \begin{pmatrix} 0&-1\\1&0 \end{pmatrix}$

Our characteristic polynomial is;

$$\lambda^2 - (a+d)\lambda + ad - bc = 0$$

In this case $\lambda^2 + 1$ as $a+d = 0$, as is $a\times d$, and $b\times c = -1$, so minus -1 gives us $+1$.

$\lambda^2 + 1 = 0$

Doesn't have any real numbered solutions at all. Hence, no real eigenvectors.

> We saw that our approach required finding the roots of a polynomial of order n, i.e., the dimension of your matrix. Which means that the problem will very quickly stop being possible by analytical methods alone. So when a computer finds the eigensolutions of a 100 dimensional problem it's forced to employ iterative numerical methods.

__Practice__

Practice calculating and solving the characteristic polynomial to find the eigenvalues of simple matrices.

__Q1__

$A = \begin{pmatrix} 1&0\\0&2 \end{pmatrix}$, what is the characteristic polynomial, and the solutions to the characteristic polynomial?

$\lambda^2 - 3\lambda + 2 = 0$

$\lambda_1 = 1, \lambda_2 = 2$

Select the eigenvectors;

$\begin{bmatrix} 0\\2 \end{bmatrix}$ $\begin{bmatrix} 0\\3 \end{bmatrix}$ $\begin{bmatrix} 1\\0 \end{bmatrix}$

> Recall that if a vector is an eigenvector of a matrix, then so is any (non-zero) multiple of that vector.

> One way to check that a vector is an eigenvector is to simply apply the matrix transformation and see if this is the same as multiplying by a scalar. Another way is to calculate the eigenvector by hand.

__Q2__

$A = \begin{pmatrix} 3&4\\0&5 \end{pmatrix}$, what is the characteristic polynomial, and the solutions to the characteristic polynomial?

$\lambda^2 - 8\lambda + 15 = 0$

$\lambda_1 = 3, \lambda_2 = 5$

Select the eigenvectors;

$\begin{bmatrix} 2\\1 \end{bmatrix}$ $\begin{bmatrix} 3\\0 \end{bmatrix}$ $\begin{bmatrix} -1\\-\frac{1}{2} \end{bmatrix}$ $\begin{bmatrix} 0\\0 \end{bmatrix}$

For example

$A \times \begin{bmatrix} 2\\1 \end{bmatrix} = \begin{bmatrix} 10\\5 \end{bmatrix}$ equivalent to scaling by 5

$A \times \begin{bmatrix} -1\\-\frac{1}{2} \end{bmatrix} = \begin{bmatrix} -5\\-2.5 \end{bmatrix}$ equivalent to scaling by 5

__Q3__

$A = \begin{pmatrix} 1&0\\-1&4 \end{pmatrix}$, what is the characteristic polynomial, and the solutions to the characteristic polynomial?

$\lambda^2 - 5\lambda + 4 = 0$

$\lambda_1 = 1, \lambda_2 = 4$

Select the eigenvectors;

$\begin{bmatrix} 3\\1 \end{bmatrix}$ $\begin{bmatrix} 0\\1 \end{bmatrix}$

__Q4__

$A = \begin{pmatrix} -3&8\\2&3 \end{pmatrix}$, what is the characteristic polynomial, and the solutions to the characteristic polynomial?

$\lambda^2 -25 = 0$

$\lambda_1 = -5, \lambda_2 = 5$

Select the eigenvectors;

$\begin{bmatrix} -1\\-1 \end{bmatrix}$ $\begin{bmatrix} 1\\1 \end{bmatrix}$ $\begin{bmatrix} 4\\-1 \end{bmatrix}$

For example

$A \times \begin{bmatrix} 4\\-1 \end{bmatrix} = \begin{bmatrix} -20\\5 \end{bmatrix}$ equivalent to scaling by 5 and inversion

In this case $\begin{bmatrix} 0\\2 \end{bmatrix}$ is not;

$A \times \begin{bmatrix} 0\\2 \end{bmatrix} = \begin{bmatrix} 10\\6 \end{bmatrix}$ not very eigen

__Q5__

$A = \begin{pmatrix} 5&4\\-4&-3 \end{pmatrix}$, what is the characteristic polynomial, and the solutions to the characteristic polynomial?

$\lambda^2 - 2\lambda  + 1 = 0$

$\lambda_1 = \lambda_2 = 1$

This matrix has one repeated eigenvalue - which means it may have one or two distinct eigenvectors (which are not scalar multiples of each other).

__Q6__

$A = \begin{pmatrix} -2&-3\\1&1 \end{pmatrix}$, what is the characteristic polynomial, and the solutions to the characteristic polynomial?

$\lambda^2 + \lambda + 1 = 0$

No real solutions

This matrix has no real eigenvalues, so any eigenvalues are complex in nature. This is beyond the scope of this course, so we won't delve too deeply on this.

### Using eigenvectors as your basis

_Diagonalisation_ is when we perform efficient matrix operations using eigenvectors as our basis.

There will be times when we want to apply the same matrix transformation multiple times. For instance, in a time series, once an event occurs.

We start at $v_0$, then apply $T$ to get to $v_1$, then apply $T$ again, to get to $v_2$

![]({{ site.baseurl }}/images/linear_algebra/eigenbasis1.png){:width="400"}

$v_2$ is equivalent to applying $T$ to $v_1$ or applying $T$ to $v_0$ twice

$$v_2 = Tv_1 = T(Tv_0) = T^2v_0$$

We can generalise this to $n$ times, $v_n = T^nv_0$

Matrix multiplication can be comuputationally expensive, particularly as the dimensions increase.

If all the terms in a matrix are zero, except for those along the leadig diagonal, we refer to it as a diagonal matrix. When raising matrices to powers, i.e. as in the example above, applying the same transformation continuously, diagonal matrices make things a lot easier.

$A = \begin{bmatrix} 2&0\\0&2 \end{bmatrix}$

$A^3 = \begin{bmatrix} 8&0\\0&8 \end{bmatrix}$

$T^n =  \begin{pmatrix} a^n&0&0\\0&b^n&0\\0&0&c^n \end{pmatrix}$

However, if $T$ is not a diagonal matrix, we transform it to an eigenbasis, which will be diagonal.

As we saw in the section on changing basis, each column of our transform matrix simply represents the new location of the transformed unit vectors.

To build our eigen-basis conversion matrix ($C$), we plug in each of our eigenvectors as columns ($ev$).

$C = \begin{pmatrix} ev_1\\ev_2\\ev_3 \end{pmatrix}$

Our diagonal matrix $D$, which is what we'll scale up, contains the corresponding eigenvalues of the matrix $T$

$T^n =  \begin{pmatrix} \lambda_1&0&0\\0&\lambda_2&0\\0&0&\lambda_3 \end{pmatrix}$

We convert our matrix $T$ to our eigenbasis, applying our diagonalised matrix and then convert back again.

$T = CDC^{-1}$

$T^2 = CDC^{-1}CDC^{-1}$

Multiplying a matrix by it's inversion directly, as we do in the middle of $T^2$ is the same as doing nothing at all, so we can remove this expression;

$T^2 = CDDC^{-1} = CD^2C^{-1}$

This becomes generalisable;

$T^n = CDDC^{-1} = CD^nC^{-1}$

We now have a method which lets us apply a transformation matrix as many times as we'd like without paying a large computational cost. 

![]({{ site.baseurl }}/images/linear_algebra/eigenbasis2.png){:width="400"}

### Eigenbasis example

Using the transformation matrix $T = \begin{pmatrix} 1&1\\0&2\end{pmatrix}$

As the first column is just `[1, 0]`, this means that our $\hat{i}$ vector will be unchanged. However, the second column tells us that $\hat{j}$, the second vector will be moving to the point `[1, 2]`. 

> This particular transform could be decomposed into a vertical scaling by a factor of 2, and then a horizontal shear by a half step.

The diagonal vector of `[1, 1]` will become `[2, 2]` working below;

```python
# T rows * columns
(1 * 1) + (1 * 1) = 2
(0 * 1) + (2 * 1) = 2
```

__Applying T__

Say we apply $T$ to `[-1, 1]`, here we are applying $T$ for the first time

$\begin{pmatrix} 1&1\\0&2\end{pmatrix} \begin{pmatrix} -1\\1\end{pmatrix} = \begin{pmatrix} -1+1\\0+2\end{pmatrix} = \begin{pmatrix} 0\\2\end{pmatrix}$

Apply $T$ again, here we are applying $T$ for the second time

$\begin{pmatrix} 1&1\\0&2\end{pmatrix} \begin{pmatrix} 0\\2\end{pmatrix} = \begin{pmatrix} 0+2\\0+4\end{pmatrix} = \begin{pmatrix} 2\\4\end{pmatrix}$

However, we could have started with finding $T^2$;

$\begin{pmatrix} 1&1\\0&2\end{pmatrix} \begin{pmatrix} 1&1\\0&2\end{pmatrix} = \begin{pmatrix} 1&3\\0&4\end{pmatrix}$

And applying that to our vector `[-1, 1]`

$\begin{pmatrix} 1&3\\0&4\end{pmatrix} \begin{pmatrix} -1\\1\end{pmatrix} = \begin{pmatrix} -1+3\\0+4\end{pmatrix} = \begin{pmatrix} 2\\4\end{pmatrix}$

We can do the whole process using our eigenbasis approach;

The conversion matrix is made up of the eigen vectors; 

eigenvectors and eigenvalues;

$ev_1 = [1, 0], \lambda = 1$, x axis vector ($\hat{i}$) does not alter

$ev_2 = [1, 1], \lambda = 2$, diagonal vector scales by 2

$C = \begin{pmatrix} 1&1\\0&1\end{pmatrix}$

The inverse of $C$ can be calculated mentally, we shift to the left, rather than the right, though best to compute computationally;

$C^{-1} = \begin{pmatrix} 1&-1\\0&1\end{pmatrix}$

$T^2 = CD^2C^{-1}$

_remember_ $D$ is the diagonal matrix of $T$

$$T^2 = \begin{pmatrix} 1&1\\0&1\end{pmatrix} \begin{pmatrix} 1&0\\0&2\end{pmatrix}^2 \begin{pmatrix} 1&-1\\0&1\end{pmatrix}$$

_start from the in, and work our way out_

$$T^2 = \begin{pmatrix} 1&1\\0&1\end{pmatrix} \begin{pmatrix} 1&-1\\0&4\end{pmatrix}$$

_one more step_

$$T^2 = \begin{pmatrix} 1&3\\0&4\end{pmatrix}$$

_then apply to our vector_ `[-1, 1]`

$$\begin{pmatrix} 1&3\\0&4\end{pmatrix} \begin{pmatrix} -1\\1\end{pmatrix} = \begin{pmatrix} 2\\4\end{pmatrix}$$

__Practice__

__Q1__

Give the matrix $T = \begin{pmatrix} 6&-1\\2&3\end{pmatrix}$ and a change basis matrix of $C = \begin{pmatrix} 1&1\\1&2\end{pmatrix}$

Calculate $D = C^{-1}TC$

$D = \begin{pmatrix} 1&1\\1&2\end{pmatrix}$

```python
# start w T * C
a = (6 * 1) + (-1 * 1) = 5
b = (6 * 1) + (-1 * 2) = 4
c = (2 * 1) + (3 * 1) = 5
d = (2 * 1) + (3 * 2) =  8

[[5, 4], [5, 8]]

# verify
np.array([[6, -1], [2, 3]]) @ np.array([[1, 1], [1, 2]])
```

$C^{-1}$ flip the off diagonal and make the leading diagonal negative; $\begin{pmatrix} 2&-1\\-1&1\end{pmatrix}$

> Recall that when a matrix is transformed into its diagonal form, the entries along the diagonal are the eigenvalues of the matrix - this can save lots of calculation!

```python
from numpy.linalg import inv

C^-1 = inv([[1, 1], [1, 2]])

c_inv @ np.array([[5, 4], [5, 8]])

array([[5., 0.],
       [0., 4.]])
```

__Q2__

Give the matrix $T = \begin{pmatrix} 2&7\\0&-1\end{pmatrix}$ and a change basis matrix of $C = \begin{pmatrix} 7&1\\-3&0\end{pmatrix}$

Calculate $D = C^{-1}TC$

```python
# start w T * C
a = (2 * 7) + (7 * -3) = -7
b = (2 * 1) + (7 * 0) = 2
c = (0 * 7) + (-1 * -3) = 3
d = (0 * 1) + (-1 * 0) = 0

[[-7, 2], [3, 0]]

# verify
np.array([[2, 7], [0, -1]]) @ np.array([[7, 1], [-3, 0]])
```

$C^{-1}$ flip the off diagonal and make the leading diagnoal negative; $\begin{pmatrix} 0&1\\-3&7\end{pmatrix}$ does not seem to work here

```python
from numpy.linalg import inv

c_inv = inv(np.array([[7, 1], [-3, 0]])) = array([[0, -0.333], [1,  2.333]])

c_inv @ np.array([[-7, 2], [3, 0]])

array([[-1., 0.],
       [4.4409, 2]]) # something wrong bottom left

# should be
array([[-1., 0.],
       [0, 2]])    
```

__Q3__

Give the matrix $T = \begin{pmatrix} 1&0\\2&-1\end{pmatrix}$ and a change basis matrix of $C = \begin{pmatrix} 1&0\\1&1\end{pmatrix}$

Calculate $D = C^{-1}TC$

```python
# start w T * C
a = (1 * 1) + (0 * 1) = 1
b = (1 * 0) + (0 * 1) = 0
c = (2 * 1) + (-1 * 1) = 1
d = (2 * 0) + (-1 * 1) = -1

[[1, 0], [1, -1]]

# verify
np.array([[1, 0], [2, -1]]) @ np.array([[1, 0], [1, 1]])
```

$C^{-1}$ flip the off diagonal and make the leading diagonal negative; $\begin{pmatrix} 1&0\\-1&1\end{pmatrix}$

```python
from numpy.linalg import inv

c_inv = inv(np.array([[1, 0], [1, 1]]))

# array([[1, 0], [-1, 1]])

c_inv @ np.array([[1, 0], [1, -1]])

array([[1, 0], [0, -1]])
```

__Q4__

Give the matrix $T = \begin{pmatrix} a&0\\0&a\end{pmatrix}$ and a change basis matrix of $C = \begin{pmatrix} 1&2\\0&1\end{pmatrix}$ and $C^{-1} = \begin{pmatrix} 1&-2\\0&1\end{pmatrix}$

Calculate $D = C^{-1}TC$

$T = \begin{pmatrix} a&0\\0&a\end{pmatrix}$

_TODO return and do properly with eigenvectors_

__Q5__

Give the matrix $T = \begin{pmatrix} 6&-1\\2&3\end{pmatrix} = \begin{pmatrix} 1&1\\1&2\end{pmatrix} \begin{pmatrix} 5&0\\0&4\end{pmatrix} \begin{pmatrix} 2&-1\\-1&1\end{pmatrix}$

Calculate the matrix $T^3$

```python
import numpy as np

np.array([[6, -1], [2, 3]]) @ 
    np.array([[6, -1], [2, 3]]) @ 
    np.array([[6, -1], [2, 3]])

[[186, -61], [122, 3]]
```

_TODO return and do properly with eigenvectors_

__Q6__

Give the matrix $T = \begin{pmatrix} 2&7\\0&-1\end{pmatrix} = \begin{pmatrix} 7&1\\-3&0\end{pmatrix} \begin{pmatrix} -1&0\\0&2\end{pmatrix} \begin{pmatrix} 0&-\frac{1}{3}\\1&\frac{7}{3}\end{pmatrix}$

Calculate the matrix $T^3$

```python
import numpy as np

np.array([[2, 7], [0, -1]]) @ 
    np.array([[2, 7], [0, -1]]) @ 
    np.array([[2, 7], [0, -1]])

[[8, 21], [0, -1]]
```

_TODO return and do properly with eigenvectors_

__Q7__

Give the matrix $T = \begin{pmatrix} 1&0\\2&-1\end{pmatrix} = \begin{pmatrix} 1&0\\1&1\end{pmatrix} \begin{pmatrix} 1&0\\0&-1\end{pmatrix} \begin{pmatrix} 1&0\\-1&1\end{pmatrix}$

Calculate the matrix $T^5$

```python
import numpy as np

np.array([[1, 0], [2, -1]]) @ 
    np.array([[1, 0], [2, -1]]) @ 
    np.array([[1, 0], [2, -1]]) @ 
    np.array([[1, 0], [2, -1]]) @ 
    np.array([[1, 0], [2, -1]])

[[1, 0], [2, -1]]
```

_TODO return and do properly with eigenvectors_

## Making the PageRank algorithm

The central assumption underpinning page rank is that the importance of a website is related to its links to and from other websites.

We're trying to build an expression that tells us, based on this network structure, which of these webpages is most relevant to the person who made the search.

![]({{ site.baseurl }}/images/linear_algebra/pagerank1.png){:width="200"}

By mapping all the possible links, we can build a model to estimate the amount of time we would expect Procrastinating Pat to spend on each webpage.

We can describe the links present on page A as a vector, where each row is either a one or a zero based on whether there is a link to the corresponding page. And then normalise the vector by the total number of the links, such that they can be used to describe a probability for that page.

$A = \begin{bmatrix}0,&1,&1,&1\end{bmatrix}$ $A$ has a link to each of $B$, $C$, and $D$ but not itsself.

We'll normalise by a third, as there are three links.

Don't forget, $L_A$ is a column in our final $L$ matrix

$L_A = \begin{bmatrix}0,&\frac{1}{3},&\frac{1}{3},&\frac{1}{3}\end{bmatrix}$

$L_B = \begin{bmatrix}\frac{1}{2},&0,&0,&\frac{1}{2}\end{bmatrix}$

$L_C = \begin{bmatrix}0,&0,&0,&1\end{bmatrix}$

$L = \begin{pmatrix}0&\frac{1}{2}&0&0 \\ \frac{1}{3}&0&0&\frac{1}{2} \\ \frac{1}{3}&0&0&\frac{1}{2} \\ \frac{1}{3}&\frac{1}{2}&1&0\end{pmatrix}$

The only way to get to $A$ is by being at $B$, which depends on being at $A$ or $D$. This problem is _self-referential_, as the ranks on all the pages depend on all the others.

> Although we built our matrix from columns of outward links, we can see that the rows describe inward links normalized with respect to their page of origin.

The vector $R$ will store the rank of all web pages.

To calculate the rank of page A, you need to know three things about all other pages on the Internet.

- What's your rank?
- Do you link to page A?
- How many outoging links do you have in total?

$r_A = \sum^n_{j=1} L_{A, j} r{j}$ 

$r_A$ is the sum of all the locations * multiplied by their rank

In the expression above the $\sum$ means from where $J = 1$ to $n$, $n$ is the number of web pages.

All positions in the link matrix relevent to webpage $A$ at location $j$, multiplied by the rank at location $j$

The rank of $A$ is the sum of all the pages linked to it, weighted by their specific link probability (taken from $L$)

We can re-write this as a matrix multiplication;

$r = Lr$

> Now as we start off not knowing $r$ we assume that all the ranks are equally and normalise them by the total number of webpages in our analysis, in this case is 4

$r = \begin{pmatrix}\frac{1}{4} \\ \frac{1}{4} \\ \frac{1}{4} \\ \frac{1}{4} \end{pmatrix}$

Each time you multiply r by our matrix L, this gives us an updated value for r

$r^{i+1} = Lr^i$

We solve the problem by iterating through the matrix $L$ until $r$ stops changing

$r$ ban be thought of as an eigenvector of $L$ with an eigenvalue of 1

We can't use the diagonal method, as it requires us knowing all the eigenvectors, which is what we're trying to calculate.

> Although there are many approaches for efficiently calculating eigenvectors that have been developed over the years, repeatedly multiplying a randomly selected initial guest vector by a matrix, which is called _the power method_, is still very effective
>
> The power method gives you one eigenvector, even though there will be _n_ for an _n_ webpage system, because of how we've structured our link matrix $L$, it will always give you an eigenvector with a value of 1 -> this will be the largest eigenvalue
>
> In an example with lots of web pages, most pages won't link to one another, there will be lots of 0s in _L_, this is referred to as a sparse matrix, allowing us to perform very efficient multiplication.

The __damping factor__ $d$ adds an additional form to our iterative formula;

$r^{i+1} = d(Lr^i) + \frac{1-d}{n}$

$d$ is a value between 0 and 1

You can think of it as 1 minus the probability with which Procrastinating Pat suddenly, randomly types in a web address, rather than clicking on a link on his current page. 

## Diving into the PageRank algorithm

The PageRank algorithm is based on an ideal random web surfer who, when reaching a page, goes to the next page by clicking on a link. The surfer has equal probability of clicking any link on the page and, when reaching a page with no links, has equal probability of moving to any other page by typing in its URL. In addition, the surfer may occasionally choose to type in a random URL instead of following the links on a page. The PageRank is the ranked order of the pages from the most to the least probable page the surfer will be viewing.

__PageRank as a linear algebra problem__
Let's imagine a micro-internet, with just 6 websites (**A**vocado, **B**ullseye, **C**atBabel, **D**romeda, **e**Tings, and **F**aceSpace). Each website links to some of the others, and this forms a network as shown,

![]({{ site.baseurl }}/images/linear_algebra/pagerank2.png){:width="400"}

Imagine we have 100 *Procrastinating Pat*s on our micro-internet, each viewing a single website at a time.
Each minute the Pats follow a link on their website to another site on the micro-internet.
After a while, the websites that are most linked to will have more Pats visiting them, and in the long run, each minute for every Pat that leaves a website, another will enter keeping the total numbers of Pats on each website constant.
The PageRank is simply the ranking of websites by how many Pats they have on them at the end of this process.

We represent the number of Pats on each website with the vector,
$$\mathbf{r} = \begin{bmatrix} r_A \\ r_B \\ r_C \\ r_D \\ r_E \\ r_F \end{bmatrix}$$
And say that the number of Pats on each website in minute $i+1$ is related to those at minute $i$ by the matrix transformation

$$ \mathbf{r}^{(i+1)} = L \,\mathbf{r}^{(i)}$$

with the matrix $L$ taking the form,
$$ L = \begin{bmatrix}
L_{A→A} & L_{B→A} & L_{C→A} & L_{D→A} & L_{E→A} & L_{F→A} \\
L_{A→B} & L_{B→B} & L_{C→B} & L_{D→B} & L_{E→B} & L_{F→B} \\
L_{A→C} & L_{B→C} & L_{C→C} & L_{D→C} & L_{E→C} & L_{F→C} \\
L_{A→D} & L_{B→D} & L_{C→D} & L_{D→D} & L_{E→D} & L_{F→D} \\
L_{A→E} & L_{B→E} & L_{C→E} & L_{D→E} & L_{E→E} & L_{F→E} \\
L_{A→F} & L_{B→F} & L_{C→F} & L_{D→F} & L_{E→F} & L_{F→F} \\
\end{bmatrix}
$$

The columns represent the probability of leaving a website for any other website, and sum to one.

The rows determine how likely you are to enter a website from any other, though these need not add to one.

The long time behaviour of this system is when $\mathbf{r}^{(i+1)} = \mathbf{r}^{(i)}$, so we'll drop the superscripts here, and that allows us to write,
$$ L \,\mathbf{r} = \mathbf{r}$$

This is an eigenvalue equation for the matrix $L$, with eigenvalue 1 (this is guaranteed by the probabalistic structure of the matrix $L$).

```python
%pylab notebook
import numpy as np
import numpy.linalg as la
from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)

# column 1 represents the possibilites of going from A to each other site, and so on
L = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   1/2 ],
              [1/3, 0,   1/3, 0, 1/2, 1/2 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]])
```

We could use a linear algebra library, as below. But this gets unmanagable for large systems. 

```python
eVals, eVecs = la.eig(L) # Gets the eigenvalues and vectors
order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues
eVals = eVals[order]
eVecs = eVecs[:,order]

r = eVecs[:, 0] # Sets r to be the principal eigenvector
100 * np.real(r / np.sum(r)) # Make this eigenvector sum to one, then multiply by 100 Procrastinating Pats

# output
array([16., 5.333, 40, 25.333, 0, 13.333])
```

the PageRank of this micro-internet is: **C**atBabel, **D**romeda, **A**vocado, **F**aceSpace, **B**ullseye, **e**Tings

Since we only care about the principal eigenvector (the one with the largest eigenvalue, which will be 1 in this case), we can use the power iteration method which will scale better, and is faster for large systems.

Let's now try to get the same result using the Power-Iteration method.

First let's set up our initial vector, $\mathbf{r}^{(0)}$, so that we have our 100 Procrastinating Pats equally distributed on each of our 6 websites.

```python
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)

# output
array([16.66666667, 
    16.66666667, 
    16.66666667,  
    16.66666667, 
    16.66666667, 
    16.66666667
])
```

Update the vector to the next minute, with the matrix  L

```python
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
for i in np.arange(100) : # Repeat 100 times
    r = L @ r

# output
array([16., 5.333, 40, 25.333, 0, 13.333])
```

Even better, we can keep running until we get to the required tolerance.

```python
r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L @ r
    i += 1
print(str(i) + " iterations to convergence.")

# output
# 18 iterations to convergence.
array([16., 5.333, 40, 25.333, 0, 13.333])
```

__Damping Parameter__

Say a new website is added to the micro-internet: *Geoff's* Website.

![]({{ site.baseurl }}/images/linear_algebra/pagerank3.png){:width="400"}

```python
L2 = np.array([[0,   1/2, 1/3, 0, 0,   0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = L2 @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L2 @ r
    i += 1
print(str(i) + " iterations to convergence.")

# output
# 131 iterations to convergence.
array([0.03046998, 0.01064323, 0.07126612, 0.04423198, 1, 0.02489342, 99.81849527])
```
This behaviour can be understood, because once a Pat get's to Geoff's Website, they can't leave, as all links head back to Geoff.

To combat this, we can add a small probability that the Procrastinating Pats don't follow any link on a webpage, but instead visit a website on the micro-internet at random. 

We'll say the probability of them following a link is $d$ and the probability of choosing a random website is therefore $1-d$.
We can use a new matrix to work out where the Pat's visit each minute.
$$ M = d \, L + \frac{1-d}{n} \, J $$
where $J$ is an $n\times n$ matrix where every element is one.

```python
d = 0.5 # Feel free to play with this parameter after running the code once.
M = d * L2 + (1-d)/7 * np.ones([7, 7]) # np.ones() is the J matrix, with ones for each entry.

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = M @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = M @ r
    i += 1
print(str(i) + " iterations to convergence.")

# output
# 8 iterations to convergence.
array([ 13.68217054, 11.20902965, 22.41964343,  16.7593433, 7.14285714, 10.87976354, 17.90719239])
```

### Produce a function that can calculate the PageRank for an arbitrarily large probability matrix

```python
import numpy as np
import numpy.linalg as la
from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)

def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    r = 100 * np.ones(n) / n
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1
    return r
```

### Eigenvalues and eigenvectors practice

`Numpy` can calculate eigenvectors and eigenvalues

```python
M = np.array([[ 4, -5,6],
       [7, -8, 6],
       [3/2,  -1/2, -2]])
vals, vecs = np.linalg.eig(M)
vecs

# output
[[ 3. -2.  1.]
 [ 3. -2. -1.]
 [ 1.  1. -2.]]
```

__Q1__ Select all the eigenvectors of the matrix above;

$\begin{bmatrix} 1/2 \\ -1/2 \\ -1\end{bmatrix}$, corresponds to the third column in the output

$\begin{bmatrix}-3 \\ -3 \\ -1\end{bmatrix}$, corresponds to the first column in the output

$\begin{bmatrix}-2/\sqrt{9} \\ -2/\sqrt{9} \\ 1/\sqrt{9}\end{bmatrix}$, corresponds to the second column in the output

__Q2__ 
In PageRank, we care about the eigenvector of the link matrix, $L$, that has eigenvalue 1, and that we can find this using power iteration method as this will be the largest eigenvalue.

PageRank can sometimes get into trouble if closed-loop structures appear. A simplified example might look like this,

![]({{ site.baseurl }}/images/linear_algebra/pagerank4.png){:width="200"}

Therefore the $L$ matrix will look like this;

$L=\left[\begin{array}{llll}
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{array}\right]$

What might be going wrong?

(1) Because of the loop, _Procrastinating Pats that are browsing will go around in a cycle rather than settling on a webpage. The system will never converge using the power iteration method.

(2) Other eigenvalues are not small compated to 1, and so do not decay away with each power iteration. The other eigenvectors in fact have the same size as 1 (they are $-1, i, -i$)

What we can do to overcome this, is to add damping

$L^{\prime}=\left[\begin{array}{cccc}
0.1 & 0.1 & 0.1 & 0.7 \\
0.7 & 0.1 & 0.1 & 0.1 \\
0.1 & 0.7 & 0.1 & 0.1 \\
0.1 & 0.1 & 0.7 & 0.1
\end{array}\right]$

There is now a probability to move to any website. Pats are no longer constrained by the loop.

_The other eigen values get smaller_

Another issue that arises, is when parts of the system are decoupled.

For example;

![]({{ site.baseurl }}/images/linear_algebra/pagerank5.png){:width="200"}

With a link matrix;

$L=\left[\begin{array}{llll}
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{array}\right]$

This form is known as block diagonal, as it can be split into square blocks along the main diagonal, i.e.

$L= \begin{bmatrix}A & 0 \\ 0 & B\end{bmatrix}$

With $A = B = \begin{bmatrix}0 & 1 \\ 1 & 0\end{bmatrix}$ in this case.

In this system, there are two loops $(A \rightleftarrows B)$ and $(C \rightleftarrows D)$.

As the system is disconnected there will not be a unique PageRank. There are two eigenvalues of 1.

The eigensystem is degenerate. Any linear combination of eigenvectors with the same eigenvalue is also an eigenvector.

The power iteration algorithm could settle on multiple values, depending on its starting conditions.

Damping in this setup will produce only one eigenvalue of 1 and PageRank will settle to its' eigenvector.

----

Good note on the differences between a vecotr, matrix and a tensor [here](https://medium.com/@quantumsteinke/whats-the-difference-between-a-matrix-and-a-tensor-4505fbdc576c#:~:text=The%20basic%20idea%2C%20though%2C%20is,of%20as%20a%20generalized%20matrix.&text=Any%20rank%2D2%20tensor%20can,really%20a%20rank%2D2%20tensor.)

## Refs

Imperial's "Mathematics for Machine Learning: Linear Algebra" on [Coursera](https://www.coursera.org/learn/linear-algebra-machine-learning/)

Joel Grus' "Data Science from Scratch" [Github](https://github.com/joelgrus/data-science-from-scratch)

3Blue1Brown's essence of linear algebra [Youtube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

Standford's CS229 [pdf](http://cs229.stanford.edu/summer2019/cs229-linalg.pdf)