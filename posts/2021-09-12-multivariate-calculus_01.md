---
toc: true
layout: post
description: Notes from Imperial's "Mathematics for Machine Learning course"
categories: [maths for machine learning]
title: Building an intuition for Calculus
---
# Calculus, a gentle introduction

Mathematical language is like any other. You can't enjoy poetry until you understand a lot of vocabulary and grammar. Don't shy away from the "boring, rote" work.

## What is calculus

The science of change

> Calculus, like other forms of mathematics, is much more than a language; it's also an incredibly powerful system of reasoning.
> 
> It lets us transform one equation into another by performing various symbolic operations on them, operations subject to certain rules. Those rules are deeply rooted in logic, so even though it might seem like we're just shuffling symbols around, we're actually constructing long chains of logical inference. The symbol shuffling is useful shorthand, a convenient way to build arguments too intricate to hold in our heads. 
>
> The truly radical and distinctive move of calculus is that it takes the divide-and-conquer strategy to its utmost extreme - all the way out to infinity. Instead of cutting a big problem into a handful of bite-size pieces, it keeps cutting and cutting relentlessly until the problem has been chopped and pulverized into its tiniest conceivable parts, leaving infinitely many of them. Once that's done, it solves the original problem for all the tiny parts, which is usually a much easier task than solving the initial giant problem.
>
> Thus, calculus proceeds in two phases: cutting and rebuilding.
> 
> (1) In mathematical terms, the cutting process always involves infinitely fine subtraction, which is used to quantify the differences between the parts. Accordingly, this half of the subject is called __differential calculus__. (2) The reassembly process always involves infinite addition, which integrates the parts back into the original whole. This half of the subject is called __integral calculus__.
> 
> This strategy can be used on anything that we can imagine slicing endlessly. Such infinitely divisible things are called continua and are said to be continuous, from the Latin roots con (together with) and tenere (hold), meaning uninterrupted or holding together.
> 
> Why should the universe respect the workings of any kind of logic, let alone the kind of logic that we puny humans can muster? This is what Einstein marvelled at, when he wrote, "The eternal mystery of the world is its comprehensibility. And it's what Eugene Wigner meant in his essay "On the Unreasonable Effectiveness of Mathematics in the Natural Sciences" when he wrote, "The miracle of the appropriateness of the language of mathematics for the formulation of the laws of physics is a wonderful gift which we neither understand nor deserve."
> 
> _Infinite Powers -- Steven Strogatz_

## Functions

Functions are a relationship between an input and an output.

If we want the temperate of a room at a given time, our inputs might be the `x, y, z` co-ordinates of the room and `t` to represent time. Our function would then return the temperature.

$$ T(x, y, z, t) = temp $$

$f(x) = x^2 + 3$, here, $f$ is a function of $x$

Without context, this can get tricky. Take the example below;

$$ f(x) = g(x) + h(x - a) $$

We can assume assume $g, h, a$ are not variables, otherwise we would have to write $f(x, g, h, a)$.

But without context it's not 100% clearly if $g$ is a function acting on $x$, are $h$ and $a$ constants, or is $h$ also a function.

Selecting a candidate function or hypothesis to model the world is the creative essence of science.

> Calculus is the study of describing the relationship between a function and the change in its variables. 

### Matching a description of a function to a graph of a function

> (1) Imagine that you place one end of a water hose into a swimming pool and turn the tap on at the other end. Water then pours into the pool at a constant rate, causing the volume of water in the pool to increase at a constant rate.
>
> While the swimming pool is still filling up with water, what would we expect the plot of the function of volume of water in the pool with respect to time to look like?

![]({{ site.baseurl }}/images/calculus/quiz1.1.png){:width="300px"}

As water flows in at a constant rate, the volume increases at a constant rate, so the graph is just a straight line. 

> (2) The tea is left to cool down. The speed of cooling depends on the temperature of the tea: when it is hot it cools down quickly and as it gets colder it cools down more and more slowly, until it approaches room temperature.
> 
> Which of the following graphs could represent the temperature of that cup of tea with time?

![]({{ site.baseurl }}/images/calculus/quiz1.2.png){:width="300px"}

Newton's law of cooling. This is an exponential function

> (3) Rahul drops a ball from the top of a ladder into a pit of sand.
>
> When the ball is released it begins to accelerate towards the ground, getting faster and faster until it hits the sand and quickly becomes stationary again. What would a plot of the speed of the ball against time look like? 

![]({{ site.baseurl }}/images/calculus/quiz1.3.png){:width="300px"}

The plot has three distinct regions. When the ball is falling it speeds up at at a constant rate of acceleration, then it suddenly decelerates and has zero speed in the sandpit.

> (4) Bags of flour labelled 1 kg from a supermarket are weighed. Most of the weights measured are very close to 1 kg, with some a little more and others a little less. Those which are further away from 1 kg are found less and less often, with almost no bags more than 100 g out.

> What might we expect the plot of frequency (i.e. how often a type of bag is found) against mass to look like?

![]({{ site.baseurl }}/images/calculus/quiz1.4.png){:width="300px"}

The weights can be approximated by a bell-curve, called the "Normal Distribution".  

> (5) A mass is attached to a string and hung from the ceiling. It is then pulled away from its natural hanging position (called equilibrium) and released, so that it swings backwards and forwards. Let's assume there is no air resistance, so that when the mass swings back it returns all the way back to where it was originally released. It completes a full swing, away and back, every 2 seconds.
> 
> What is a reasonable plot for the displacement of the mass from equilibrium with respect to time?

![]({{ site.baseurl }}/images/calculus/quiz1.5.png){:width="300px"}

This is called a _simple harmonic oscillator_ - that is, we model the movement of the pendulum through time as a simple sine wave, with some amplitude (determined by the maximum distance of the pendulum to the equilibrium point) and some frequency (determined by the period of the swing). 

The pendulum takes 2 seconds to complete a full revolution, which can also be described as swinging at a frequency of 0.5 Hz.

****

## Gradients

We can calculate the gradient of a straight line by looking at how much the function $f(x)$ changes, divided by the amount the variable (x) changes.

This is often referred to as _"rise over run"_.

![]({{ site.baseurl }}/images/calculus/rise_over_run.png){:width="300px"}

A classic example is graph showing the speed of a car over time.

![]({{ site.baseurl }}/images/calculus/car1.png){:width="300px"}

We can see the graph starts with an initial speed of 0, it accelerates, slows, speeds up again before coming to a sharp stop.

A constant speed would be represented as a flat line.

Acceleration is a function of time in our example.

> Acceleration can be defined as the local gradient (the gradient at a single point) of a speed-time graph

We illustrate the gradient at a single point but drawing a tangent line. The line will be the same gradient of the curve at that point.

![]({{ site.baseurl }}/images/calculus/car2.png){:width="300px"}

We can plot an entirely new graph which shows us acceleration against time, rather than speed, bt recording the slop of these tangent lines at each point.

![]({{ site.baseurl }}/images/calculus/car3.png){:width="300px"}

An acceleration of 0 is equivalent a constant speed, so a flat line in the speed-time graph.

Speed is, $\frac{distance}{time}$ 

Acceleration is, $\frac{distance}{time^2}$

Velocity is the amount of distance travelled during the amount of time, $\frac{s}{t}$. Acceleration is a change of velocity over time $\frac{\Delta v}{\Delta t}$ (this is why we square time)

> What we've just done by eye is the essence of calculus. 
>
> We took a continuous function and described its slope at every point by constructing a new function, which is its derivative. 
> 
> We can in principle plot the derivative of the acceleration function following the same procedure, where we simply take the slope of the acceleration function at every point. This is the rate of change of acceleration which we can also think of as being the second derivative of the speed, and it's actually referred to as the jerk of the car.

![]({{ site.baseurl }}/images/calculus/car4.png){:width="300px"}

We can also follow the inverse procedure. Take our baseline speed function, and imagine what function this would have been the gradient of. This is the __anti-derivative__. 

In our example this would be the distance of the car from its starting position, or `displacement`.

![]({{ site.baseurl }}/images/calculus/car5.png){:width="300px"}

- The anti-derivative is closely related to the `integral`
    - the fundamental object of calculus corresponding to summing infinitesimal pieces to find the content of a continuous region [[ref](https://mathworld.wolfram.com/Integral.html)].


### Matching the graph of a function to the graph of its derivative

(1) Estimate the gradient of the tangent to the non-linear, green function at the point $(4, 2)$

![]({{ site.baseurl }}/images/calculus/quiz2-1.png){:width="300px"}

The run goes from 0 to 6, the rise goes from 2 to 8. Therefore, rise over run is $\frac{6}{6} = 1$

(2) It's possible to have a reasonable guess at what the derivative of a function will look like by considering regions of the function with different gradients.

Take the following example, we can see that there are three types of behaviour we might see in the gradient of a smooth function.

![]({{ site.baseurl }}/images/calculus/quiz2-2a.png){:width="300px"}

- On the left the function is increasing, so we have a positive gradient
- At the turning point the gradient will be exactly $0$
- On the right the function is decreasing, so we will have a negative gradient

![]({{ site.baseurl }}/images/calculus/quiz2-2b.png){:width="300px"}

> The derivative starts positive in the "increasing region" and becomes negative in the "decreasing region", passing through zero at the "turning point"

(3) What representation would best describe the following function?

![]({{ site.baseurl }}/images/calculus/quiz2-3a.png){:width="300px"}

We can see there are three turning points, so the derivative will be 0 three times.

![]({{ site.baseurl }}/images/calculus/quiz2-3b.png){:width="300px"}

(4) What representation would best describe the following function?

![]({{ site.baseurl }}/images/calculus/quiz2-4a.png){:width="300px"}

Again, we can see there are three turning points, so the derivative will be 0 three times.

![]({{ site.baseurl }}/images/calculus/quiz2-4b.png){:width="300px"}

(3) and (4) have the same derivative. Shifting a function up or down does not change the gradient.

(5) Reverse problem. What is the anti-derivative of the function below?

![]({{ site.baseurl }}/images/calculus/quiz2-5a.png){:width="300px"}

We can see there are four points at which the derivative is 0, so there are four turning points.

Both these images could be represented by the derivative plot above.

> If one function is a vertical shift of another function, then they have the same differential.

![]({{ site.baseurl }}/images/calculus/quiz2-5b.png){:width="500"}

## Derivatives

_Rise over run_ is more formally expressed as, "the gradient of a line is equal to the amount of that function changes in this interval, divided by the length of the interval we're considering".

$$ gradient = \frac{rise}{run} $$

This helps us built the intuition that 

- flat horizontal lines have a gradient of 0
- upward sloping lines have a positive gradient
- downward sloping lines have a negative gradient

A linear function will have a flat gradient, because no matter the interval (run) we're considering the slope is the same. However, for more complex functions, where the slope is constantly changing, i.e. the gradient is different at different points, the rise over run depends on where we choose our points.

![]({{ site.baseurl }}/images/calculus/derivative1.png){:width="500"}

__Important to note__. For continuous functions, as delta x $\Delta x$ gets smaller and smaller it becomes a better approximation for the gradient at point $x$. 

This goes back to what Storgatz called `the Infinity Principle`. 

> To analyze something complicated, the Infinity Principle says you should first break it down into an infinity of simpler parts and analyze those. Putting those infinitely many analysed parts back together into an analysed whole can be difficult, but it can be easier than analysing the complicated whole directly. [[ref](https://lareviewofbooks.org/article/to-infinity-and-beyond-the-power-of-calculus/)]

__Delta x is infinitely small but non-zero.__

We can express this concept formally, by using the limit notation scheme:

> as delta goes to zero, our expression will give us a function for our gradient at any point we choose

Depending on the notation schema you prefer you can write `f dash` or `f prime`: $f'$ or `df by dx`: $\frac{df}{dx}$

![]({{ site.baseurl }}/images/calculus/derivative2.png){:width="500"}

This is what it means to __differentiate a function__. You substitute the function into this expression.

$$ f'(x) = lim_{\Delta x \to 0} \left(\frac{f(x + \Delta x) - f(x)}{\Delta x}\right)$$

Differentiation is the process of finding a derivative.
### In practice, linear

We know a linear function will give us a constant derivative, so lets start there.

Our linear function: $ f(x) = 3x + 2 $

Substitute in:

$$ f'(x) = lim_{\Delta x \to 0} \left(\frac{3(x + \Delta x) +2 - (3x + 2)}{\Delta x}\right)$$

Do some algebra;

- start by expanding the brackets

$$ lim_{\Delta x \to 0} \left(\frac{3x + 3\Delta x +2 - 3x - 2}{\Delta x}\right)$$

$3x$ and $-3x$ cancel each other out, as do the $2$ and $-2$

$$ lim_{\Delta x \to 0} \left(\frac{3\Delta x}{\Delta x}\right)$$

The $\Delta x$ themselves cancel out, so we end up with a limit of 3;

$$ lim_{\Delta x \to 0} (3)$$

Because all the $\Delta x$ terms have been removed (solved for), the limit expression has no effect, so we can just ignore it.

The gradient of our linear function is a constant.

We know that gradient of a linear function $ax + b$ is just $a$ and now we see why.

### The sum rule

In the example above, we differentiated $3x$ and $+2$ as the same time. We could do this things separately and then added the results.

> This interchangeability of the approach is called The Sum Rule

$$ \frac{d}{dx} \left(f(x) + g(x)\right) = \frac{df(x)}{dx} + \frac{dg(x)}{dx} $$

Remember $\frac{df(x)}{dx}$ is $\Delta x$

### In practice, non-linear

Take the function $f(x) = 5x^2$

_Note_ Remember the square of a sum

$$ (x + \Delta x)^2 = x^2 + 2x\Delta x + (\Delta x)^2$$

Substitute in:

$$ lim_{\Delta x \to 0} \left(\frac{5(x + \Delta x)^2 - (5x^2)}{\Delta x}\right)$$

Becomes

$$ lim_{\Delta x \to 0} \left(\frac{5x^2 + 10x\Delta x + 5\Delta x^2 - 5x^2}{\Delta x}\right)$$

- $5x^2$ and $-5x^2$ cancel out
- remove a $\Delta x$ from both top terms

$$ lim_{\Delta x \to 0} \left(10x + 5\Delta x\right)$$

Because $\Delta x$ is going to become infinitesimally small, i.e. going to 0, we can forget about it, so we get $10x$

$$ f'(x) = lim_{\Delta x \to 0} (10x) $$

__The derivative of the function $5x^2$ is $10x$

### The power rule

The power rule allows us to differentiate polynomials quickly

If $f(x) = ax^b$

Then $f'(x) = abx^{b-1}$

In our example above, $(5*2)^1 = 10$

![]({{ site.baseurl }}/images/calculus/derivative3.png){:width="300"}

Rules like the `power rule` and the `sum rule` help speed up the process of differentiation. There are many of these in calculus.

### Differentiation special cases

Special case functions that give interesting results when differentiated.

### (1) $ f(x) = \frac{1}{x} $

![]({{ site.baseurl }}/images/calculus/spec_func1.svg)

The gradient of this function is negative everywhere, except at the point, x = 0. 

We can't divide by 0 so we can't see this point. 

As we move right along the graph we see there is a break in our continuous function. $f(x)$ drops towards negative infinity and then re-emerges on the positive side.

This sudden break is what is known as a _discontinuity_. This function doesn't have a value at the point $x=0$.

What about the gradient? Let's differentiate our function to find out;

$$ f'(x) = lim_{\Delta x \to 0} \left(\frac{\frac{1}{x + \Delta x} - \frac{1}{x}}{\Delta x}\right) $$

- First we need to combine our numerator into a single fraction, making the denominator of each the same.

$$ lim_{\Delta x \to 0} \left(\frac{\frac{x}{x(x + \Delta x)} - \frac{x + \Delta x}{x(x + \Delta x)}}{\Delta x}\right) $$

- The $x$ and $-x$ on our new numerator cancel out

$$ lim_{\Delta x \to 0} \left(\frac{\frac{-\Delta x}{x(x + \Delta x)}}{\Delta x}\right) $$

- The $-\Delta x$ and $\Delta x$ cancel out, giving us

$$ lim_{\Delta x \to 0} \left(\frac{-1}{x(x + \Delta x)}\right) $$

$$ lim_{\Delta x \to 0} \left(\frac{-1}{x^2 + x \Delta x}\right) $$

We now have the magic of limits to remove the term $x \Delta x$, leaving us with;

$$ \frac{-1}{x^2} $$

![]({{ site.baseurl }}/images/calculus/spec_func2.png)

This derivative function is negative everywhere and undefined at $x = 0$

### (2) $ f(x) = e^x $

In this function, the value of the function is always equal to its gradient.

$$ f(x) = f'(x) $$

This is also true for $f(x) = 0$

Our function must always be negative or always be negative. Crossing the horizontal function would cause $f'(x)$ to be 0. This property of always decreasing or always increasing also means it can never return to the same value again.

Plenty of functions would fit this in the positive direction. But only Euler's number $e$, works in both positive and negative direction.

$e = 2.71828...$

$$ f(x) = e^x = f'(x) $$

$e$, like $\pi$ appears to be written all over the fabric of the universe.

> As differentiating e to the x gives us e to the x, clearly, we can just keep differentiating this thing as many times as we'd like and nothing is going to change. This self similarity is going to come in very handy.

![]({{ site.baseurl }}/images/calculus/spec_func3.png)

### (3) Trigonometric functions, `Sine` and `Cosine`

> You may recall that for a right angled triangle, sine of angle x multiplied by the hypotenuse r gives you the length of the opposite side to the angle. 

![]({{ site.baseurl }}/images/calculus/spec_func4.png)

$Cosine(x)$ is the derivative of $Sin(x)$

![]({{ site.baseurl }}/images/calculus/spec_func5.png)

If we differentiate $cos(x)$ we get $-sin(x)$

![]({{ site.baseurl }}/images/calculus/spec_func6.png)

Differentiating $sin(x)$ four times, gives us $sin(x)$

![]({{ site.baseurl }}/images/calculus/spec_func7.png)

The pattern repeats, again giving us a property of self-similarity.

That is because trigonometric functions are exponential functions in disguise.

$$ sin(x) = \frac{e^{ix} - e^{-ix}}{2i} $$

__Differentiation may feel complicated at times, but the concept is simple. You are looking for the _rise over run_ gradient at each point.__

## Practice

__(1)__ Using the power rule, $\frac{d}{dx} \left(ax^b\right) = abx^{b-1}$

Differentiate $f(x) = x^{173}$

$f'(x) = 173x^{172}$

__(2)__ Using the sum rule, $\frac{d}{dx}\left[f(x) + g(x)\right] = \frac{df(x)}{dx} + \frac{gf(x)}{dx}$

Differentiate $f(x) = x^2 + 7 + \frac{1}{x}$

We know 

- $x^2$ becomes $2x$, 
- $7$ is a constant so we ignore it (remember the gradient of a line function $x = ab + c$, is $a$), 
- and $\frac{1}{x}$ we've been given

$f'(x) = 2x - \frac{1}{x^2}$

__(3)__ Find the second derivative of $f(x) = e^x + 2sin(x) + x^3$

We know 

- $f''(x)$ of $e^x$ is $e^x$
- $f'(x)$ of $sin(x)$ is $cos(x)$ and $f'(x)$ of that is $-sin(x)$
- $x^3$ we can use the power rule: $x^3 \to 3x^2 \to 6x$

$f''(x) = e^x + -2sin(x) + 6x$

__(4)__ To find the anti-derivative we ask what function you'd need to differentiate to get $f'(x)$. Consider the power rule in reverse. The anti-derivative of $abx^{b-1} is ax^b$

What is the anti-derivative of $f'(x) = x^4 - sin(x) - 3e^x$?

- $x^4$ becomes $\frac{1}{5} x ^ 5$, 
    - tip: 5 * 0.2 = 1
- $-sin(x)$ becomes $cos(x)$
- $3e^x$ is $3e^x$

$ f(x) = \frac{1}{5} x ^ 5 + cos(x) - 3e^x + c$

When calculating anti-derivatives we can add any constant, since the derivative of a constant is zero.

__(5)__ The power rule can be applied for any real value of $b$

Using the facts that $\sqrt{x} = x^{\frac{1}{2}}$ and $x^{-a} = \frac{1}{x^a}$

> Power rule: multiply by the power, then reduce the power by one

Calculate $\frac{d}{dx}\left(\sqrt{x}\right)$

- Rewrite: $(x^{\frac{1}{2}})$
- Power law: $(\frac{1}{2}x^{-\frac{1}{2}})$
- Second fact given: $(\frac{1}{2}\frac{1}{x^\frac{1}{2}})$
- Becomes: $(\frac{1}{2}\frac{1}{\sqrt{x}})$
- Becomes: $(\frac{1}{2\sqrt{x}})$

$\frac{d}{dx}\left(\sqrt{x}\right) = \frac{1}{2\sqrt{x}}$

## Time saving rules

### Product rule

A convenient shortcut for differentiating the product of two functions.

Say we have a rectangle. The length of one side is the function $f(x)$ and the other $g(x)$. The product of these two function with give us the rectangle's area, $A(x) = f(x)g(x)$.

If we differentiate $A(x)$, what we're looking for is the change in area of our rectangle as we vary $x$.

> What is the derivative of A with respect to x?

![]({{ site.baseurl }}/images/calculus/product_rule1.png){:width="500"}

_Note:_ In the example chosen, both our functions increase with $x$. This does not need to be the case for the product rule to be applied.

$\Delta A(x)$ can be viewed as the sum of three triangles, each representing the change in height, width and diagonal

![]({{ site.baseurl }}/images/calculus/product_rule2.png){:width="500"}

As $\Delta x$ goes to 0, it is the smallest rectangle that will shrink the fastest. We simply ignore its contribution to the area.

Our rise over run, differentiation calculation is the limit of $\frac{\Delta A(x)}{\Delta x}$

$$ lim_{\Delta x \to 0} \left(\frac{\Delta A(x)}{\Delta x}\right) = lim_{\Delta x \to 0} \left(\frac{f(x)(g(x + \Delta x) - g(x)) + g(x)(f(x + \Delta x) - f(x))}{\Delta x}\right) $$

We can rearrange this equation. (1) Splitting into two fractions and (2) Removing $f(x)$ and $g(x)$ out of the numerators.

$$ lim_{\Delta x \to 0} \left(\frac{\Delta A(x)}{\Delta x}\right) = lim_{\Delta x \to 0} \left(f(x)\frac{(g(x + \Delta x) - g(x))}{\Delta x} + g(x)\frac{(f(x + \Delta x) - f(x))}{\Delta x}\right) $$

This first fraction is now the derivative of $g(x)$ and the second is the derivative of $f(x)$

$$ A'(x) = f(x)g'(x) + g(x)f'(x) $$

![]({{ site.baseurl }}/images/calculus/product_rule3.png){:width="500"}

![]({{ site.baseurl }}/images/calculus/product_rule4.png){:width="300"}

#### Practice the product rule

__(1)__ Write the product rule $A'(x) = f'(x)g(x) + f(x)g'(x)$ in $\frac{d}{dx}$ notation

$$ \frac{dA(x)}{dx} = \frac{df(x)}{dx} g(x) + f(x) \frac{dg(x)}{dx}$$

__(2)__ It can be useful to rewrite a function $f(x)$ into two parts so that we can apply the product rule and differentiate more easily.

$A(x) = (x + 2)(3x - 3)$ can be thought of as $f(x) = (x + 2) and g(x) = (3x - 3)$

Remember we can ignore constants, so it makes it easy to differentiate

Therefore $A'(x) = 3(x + 2) + (3x - 3) = 6x + 3$

__(3)__ Combining previous rules and our knowledge of our trigonometric functions differentiate it we can easily differentiate the function $f(x) = x^3sin(x)$

$f'(x) = 3x^2sin(x) + x^3cos(x)$

__(4)__ Differentiate $f(x) = \frac{e^x}{x}$

_Note_ $\frac{e^x}{x} = e^x\frac{1}{x}$ so can use the product rule

$f'(x) = e^x (-\frac{1}{x^2}) + e^x \frac{1}{x} = e^x \left(\frac{1}{x} - \frac{1}{x^2}\right)$

__(5)__ We can extend the product rule to the products of more than two functions

- Consider the function $u(x) = f(x)g(x)h(x)$
- Then use the product rule twice to find the expression for $u'(x)$
- This is the product rule for a product of three functions

Start by substituting A(x) = f(x)g(x)

$u'(x) = A(x)h'(x) + h(x)A'(x)$

Now substitute out (A)

$u'(x) = f(x)g(x)h'(x) + h(x)\left(f(x)g'(x) + g(x)f'(x)\right)$

Multiply out $h(x)A'(x)$ on the right hand-side

$u'(x) = f(x)g(x)h'(x) + h(x)(f(x)g'(x) + h(x)g(x)f'(x)$

__(6)__ Now differentiate a function that is the product of three functions: $f(x) = x e^x cos(x)$

Substitute $t(x) = x e^x$

$f'(x) = t(x)-sin(x) + cos(x)t'(x)$

$t'(x) = (e^x x+1) + (e^x x)$ -- _note_ the derivative of $x$ is 1

$f'(x) = e^x[(x + 1) cos(x) + x sin(x)]$

### Chain rule

Functions can be used as the input to other functions.

Consider the nested (or composite) function, `h of p of m` $h(p(m))$

We are relating the concept of $m$ to $h$ via the concept of $p$. We are chaining concepts together.

In our example, let's say $h$ is happiness, which is a function of the number of pizzas we eat $p$, which itself is a function of the money we have $m$.

$$h(p) = -\frac{1}{3}p^2 + p + \frac{1}{5}$$

This may look daunting but can be plotted as;

![]({{ site.baseurl }}/images/calculus/chain_rule1.png){:width="300"}

Without any pizza it is still possible to be happy. There is a point where some amount of pizza peaks our happiness, then it decreases until the amount of pizza consumed begins to negatively effect our happiness.

Pizza and money has a straight forward exponential relationship.

$$ p(m) = e^m - 1$$

> What we'd like to know is, by considering how much money I have now, how much effort should I put into making more, if my aim is to be happy? 
> 
> To work this out, we're going to need to know what the rate of change of happiness is, with respect to money,

This is $ \frac{dh}{dm}$.

We could directly substitute these two functions in:

$$ h(p(m)) = -\frac{1}{3}(e^m - 1)^2 + (e^m - 1) + \frac{1}{5}$$

Then we can differentiate this directly to

$$ \frac{dh}{dm} = \frac{1}{3}e^m(5 - 2e^m)$$

The chain rules allows us to handle more complex examples where direct substitution may not be feasible.

Consider the derivative of $h$ with respect to $p$, $\frac{dh}{dp}$, and of $p$ with respect to $m$ $\frac{dp}{dm}$.

In this notation convention, the derivatives are represented as quotients (a result obtained by dividing one quantity by another). The product of these two will give you the desired function $\frac{dh}{dm}$

$$ \frac{dh}{dp} * \frac{dp}{dm} = \frac{dh}{dm}$$

![]({{ site.baseurl }}/images/calculus/chain_rule3.png){:width="300"}

Take our example.

We don't want $p$ to appear in our final output, so we substitute $p$ in terms of $m$

![]({{ site.baseurl }}/images/calculus/chain_rule5.png){:width="500"}

We can see the benefit of getting more money (the dotted white line) decreases dramatically, once you have enough pizza.

![]({{ site.baseurl }}/images/calculus/chain_rule4.png){:width="500"}

> What's magic about the chain rule is that for some real-world applications, we may not have a nice analytical expression for our function. But we may still have the derivatives. So, being able to simply combine them with the chain rule becomes very powerful, indeed.

_Common mistake_: Not recognising whether a function is composite or not.

For example: $cos^2(x)$ is shorthand for $[cos(x)]^2$

> Described verbally, if $f(x) = g(h(x))$
>
> the rules says that the derivative of the composite function $f'(x)$, is 
> - the inner function $h$ within the derivative of the outer function $g'$
>
> $f'(x) = g'(h(x)) * h'(x)$

#### Practice the chain rule

The chain rule allows us to differentiate functions of functions. 

__(1)__ If $f(x) = g(h(x))$ express $f'(x)$ in terms of the chain rule notation

$$ \frac{dg}{dx} = \frac{dg}{dh} \frac{dh}{dx} = f'(x) = g'(h(x)) \cdot h'(x)$$

__(2)__ Much like the product rule, the art of the chain rule lies in identifying the components of the function that allow you to apply the rule.

Consider the function $f(x) = e^{x^2 - 3}$

This can be broken down to, $f(x) = g(h(x))$, 

Where 

- $g(h) = e^h \to g'(h) = e^{x^2-3}$
- $h(x) = x^2-3 \to h'(x) = 2x$

$$ f'(x) = \frac{dg}{dh} \frac{dh}{dx} = 2xe^{x^2 - 3} $$

__(3)__ Following the same process, given $f(x) = sin^3(x)$

This is shorthand for $[sin(x)]^3$

$g(h) = h^3 \to g'(h) = 3sin^2(x)$

$h(x) = sin(x) \to h'(x) = cos(x)$

$f'(x) = 3sin^2(x)cos(x)$

__(4)__ Calculate the derivative of $tan(x)$ with respect to $x$

_Note_ 

- $tan(x) = \frac{sin(x)}{cos(x)}$
- $\frac{1}{cos(x)} = [cos(x)]^{-1}$

$1 + tan^2(x) = \frac{1}{cos^2(x)}$

$$\frac{d}{dx} tan(x) = 1 + tan^2(x)$$

__(5)__ The chain rule can also be applied to functions of functions of functions.

Consider a function $f(g(h(x)))$

By applying the chain rule twice, it's possible to show that $\frac{df}{dx} = \frac{df}{dg}\frac{dg}{dh}\frac{dh}{dx}$ 

Use this to find the derivative of $f(x) = e^{sin(x^2)}$

- $f(g) = e^g \to f'(g) = e^{sin(x^2)}$
- $g(h) = sin^h \to g'(h) = cos(x^2)$
- $h(x) = x^2 \to h'(x) = 2x$

$$ f'(x) = 2xe^{sin(x^2)} cos(x^2)$$

## Combining all four rules

In this example, we'll use a complex formula

$$ f(x) = \frac{sin(2x^5 + 3x)}{e^{7x}} $$

We need to break this down into manageable pieces that allow us to apply the rules we have learnt.

(1) Though expressed as a fraction, we can rewrite our function as a product, by raising the denominator to the power of -1

$$ f(x) = (sin(2x^5 + 3x))e^{-7x} $$

We can now split $f(x)$ into the two parts $g(x)$ and $h(x)$

$$ g(x) = sin(2x^5 + 3x) $$

and 

$$ h(x) = e^{-7x}$$

Let's find an expression for $g'(x)$:

Here we have the trigonometric function $sin$ applied to a polynomial. This is a classic target for the `chain rule`.

$$ g(u) = sin(u) \to g'(u) = cos(u) $$

$$ u(x) = 2x^5 + 3x\to u'(x) = 10x^4 + 3 $$ 

$$ \frac{dg}{du} \frac{du}{dx} = cos(u)(10x^4 + 3)$$

We want a final expression that doesn't include a $u$. 

The two $\frac{du}{du}$s cancel each other out and substitute out $u$.

$$ \frac{dg}{dx} = cos(2x^5 + 3x)(10x^4 + 3) $$

Let's find an expression for $h'(x)$:

$$ h(v) = e^v \to h'(v) = e^v$$

$$ v(x) = -7x \to v'(x) = -7$$

$$ \frac{dh}{dv} \frac{dv}{dx} = -7e^{-7x}$$

So we have

$$ f'(x) = (g'(x) * h) + (h'(x) * g) $$

Where

$$ f'(x) = (cos(2x^5 + 3x)(10x^4 + 3)e^{-7x}) +  ((-7e^{-7x})sin(2x^5 + 3x)) $$

This can be rearranged and factorised

$$ f'(x) = e^{-7x} [(10x^4 + 3) cos(2x^5 + 3x) - 7sin(2x^5 + 3x)] $$ 

Or expressed in terms of the original function

$$ f'(x) = \frac{(10x^4 + 3) cos(2x^5 + 3x)}{e^{-7x}} - 7f(x)$$ 

__Premature optimisation is the root of all evil__ 

> Don't spend time tidying things up and rearranging them until that you're sure that you've finished making a mess. 

### More practice

__(1)__ $f(x) = x^{3/2} + \pi x ^2 + \sqrt{7}$

What is the derivative of $f(x)$ at the point $x = 2$?

- $x^{3/2} \to \frac{3}{2} x^{\frac{1}{2}} = \frac{3\sqrt{2}}{2}$
- $\pi x^2 \to 2\pi \cdot 2 = 4\pi$

$f'(x) = \frac{3\sqrt{2}}{2} + 4\pi$

__(2)__ What is the derivative of $f(x) = x^3cos(x)e^x$?

_Note_ product rule of three terms is  $f'(x) g(x) h(x) + f(x) g'(x) h(x) + f(x) g(x) h'(x)$

- $3x^2 cos(x) e^x = 3e^xx^2cos(x)$
- $-sin(x) x^3 e^x$
- $x^3 cos(x) e^x$

$f'(x) = 3e^xx^2cos(x) -sin(x) x^3 e^x + x^3 cos(x) e^x$

__(3)__ What is the derivative of the function $f(x) = e^{[(x+1)^2]}$?

- $g(h) = e^h$
- $h(x) = (x + 1)^2$

$f'(x) = e^{[(x+1)^2]} \cdot 2(x + 1)$

__(4)__ What is the derivative of the function $f(x) = x^2cos(x^3)$?

Product rule and chain rule on the second term

Term one

- $2x cos(x^3)$

Term two, first apply the chain rule

- $g(h) = cos(h) \to -sin(x^3)$
- $h(x) = x^3 \to 3x + 2$
- $-sin(x^3) \cdot 3x + 2$

Apply the product rule

- $-sin(x^3) \cdot 3x + 2 \cdot 2x$

TODO: check $ f'(x) = 2x cos(x^3) - sin(x^3) \cdot 3x + 2 \cdot 2x$

$ f'(x) = 2x cos(x^3) -  3x^4 sin(x^3)$

__(5)__ What is the derivative of the function $f(x) = sin(x)e^{cos(x)}$ at the point $x = \pi$

Term one, product rule

- $cos(x)e^{cos(x)}$
- note $cos(\pi) = -1$
- $-1e^{-1}$

Term two, first apply chain rule
- $g(h) = e^h$
- $h(x) = cos(x)$ 
- $e^{cos(x)}-sin(x)$

Apply the product rule

- $e^{cos(x)}-sin(x) \cdot sin(x) = e^{-1}-sin(x) \cdot sin(x)$

Bring both terms together

$-1e^{-1} + e^{-1}-sin(x) \cdot sin(x)$

- note $sin(\pi) = 0$ 

So becomes: $-1e^{-1} = -\frac{1}{e}$
