---
toc: true
layout: post
description: Notes from brilliant.org's pre-calculus course
categories: [maths for machine learning]
title: Pre-Calculus
---
## Getting started

> Master the fundamentals of exponential, logarithmic, hyperbolic, and parametric equations.

## Logarithms

The logarithm is the inverse of the exponential.

![]({{ site.baseurl }}/images/pre_calculus/log1.png){:width="300px"}

You can read $\log_b(x)$ as "$b$  raised to what power equals $x$?"

{% include info.html text="The log base 10 of a number (rounded down) is 1 less than the number of digits of the number" %}

For example; 

$\log_{10}(1000) = 3$

$\log_{10}(49323) \approx 4.69 \rightarrow\left \lfloor{\log_{10}(49323)}\right \rfloor  = 4$

$\log_{10}(333333) \approx 5.52 \rightarrow\left \lfloor{\log_{10}(333333)}\right \rfloor  = 5$

{% include info.html text="In general, every positive number with 1 followed by only 00s will have an integer answer when taking the logarithm. Adding a 0 will 'step' to the next integer." %}

$\log_{10}100=2 \;\; \log_{10}1000=3 \;\; \log_{10}10000=4$

****

With a logarithmic scale, moving forward a step multiplies the distance by a set amount. So For base 10; the scale, started a 0, i.e. $10^0 = 1. Each step is akin to multiplying the distance by 10. Moving right a _linear_ amount $x$ causes us to multiply the distance by $10^x$.

$10^2 * 10^3 = 10 ^ {(log_{10}(10^2) + log_{10}(10^3))}$

$\log_{10}(ab) = log_{10}(a) + log_{10}(b)$

****

## Exponential equations

> The rate of growth is proportional to the current amount at any given time.

$$ y = a + b^x $$

> It is impossible to undo the effect of doubling $b$ with a change in $a$

Algebraically, if we start off with $y = a \times b^xy=a×b^x$ then doubling $b$ gives $y = a \times (2b)^x = a \times 2^x \times b^x$. 

So each point is being multiplied by an additional $2^x$  term. 

This can't be undone by changing $a$ because the value of $2^x$  changes as $x$ changes while the value of $a$ will remain constant. 

So while it is possible to change $a$ so that the effect of doubling $b$ is undone for a single point it will not work for all other points.

****

## Changing the base

$$ y = b^x $$

For all bases where $b > 0$, all exponential functions pass through the point [0, 1]

![]({{ site.baseurl }}/images/pre_calculus/exp_base1.png){:width="300px"}
![]({{ site.baseurl }}/images/pre_calculus/exp_base2.png){:width="300px"}

As $b$ grows, the proportion of the graph for $x > 0$ becomes steeper, leading to a right angle shape.

The graph can be reflected over the y axis when $b$ is below 1. $y = 0.5^x$ reflects $y = 2^x$

{% include info.html text="An exponential function can't be defined with a negative base" %}

![]({{ site.baseurl }}/images/pre_calculus/exp_base3.png){:width="300px"}

Changing $b$ is the only thing that will change the shape of the curve. The others shift or scale the graph.

****

## Exponential arithmetic

$$ x^2 x^3 x^4 = x^9 $$

This is an example of the `product rule`.

{% include info.html text="these rules are true if $a$ is positive, and $m$ and $n$ are [real numbers](https://brilliant.org/wiki/real-numbers/)." %}

![]({{ site.baseurl }}/images/pre_calculus/exp_rules.png)

If we have an expontential equation, $y = ab^{x +c}$, only changing $b$ will change the shape of the curve. Changing $x$ or $c$ changes where the curve falls/steepens.

An implication of the product rule for exponential expressions;

> muliplication or division in one part of the expression can have the same effect as addition or subtraction in another part of the expression
>
> compressing or expanding the graph vertically (multiplying or dividing the entire expression) has the same effect as shifting the graph horizontally (adding or subtracting from the exponent).

Increasing $c$ stretches the graph vertically, increasing $a$, by a larger factor, will have the same effect.

$$ y = 9 \times 3^x $$

Can be rewritten using the product rule

$$ y = 3^2 \times 3^2 = 3^{x +2} $$

Exponential functions can be rewritten as logarithmic functions

$$ b^a = c \equiv log_b(c) = a $$

![]({{ site.baseurl }}/images/pre_calculus/defining_logs1.png)

****

## Graphing Logarithms

All basic logarithmic function have the same general shape.

Asking the value of $log_{10}(x)$ is equivalent to asking "what value exponeont does 10 need to be raised to in order to get $x$?

Think of the inverse function, $log_{10}(x) \equiv y = 10^x$

> The domain of $log_{10}(x)$ will be equal to the range of $y = 10^x$

Domain being the complete set of possible values of x.

All basic loagrithmic functions of the form $f(x) = log_b(x)$ have an "anchor point" - a point that they all pass through, regardless of their base.

Regardless of the base, $b$, we always have $log_b(1) = 0$ because $b^0 = 1$ for any $b$. That means all logarithmic functions will pass through the point where `x = 1 and y = 0, or (1,0)`

{% include info.html text="If $x$ is less than 1, $x$ we need to be raised to a negative exponent.

For example, with base 10, $10^{-3} = \frac{1}{10^3}$

As $x$ gets larger, the rate at which $\log_{10}(x)$ increases will become slower and slower" %}

![]({{ site.baseurl }}/images/pre_calculus/graphing_logs1.png)

The graph of $y = \log_{10}(x)$ is a reflection of the graph $y = 1-^x$ over the line $y=x$. This is a result of the inverse relationship between the two functions.

![]({{ site.baseurl }}/images/pre_calculus/graphing_logs2.png)

> It is not possible to define $\log_1(x)$ because $y = 1^x$ has the same output for any value of $x$, $1^{60}$, is the same as $1^{900}$.

****

## Understanding logarithmic arithmetic

$$ 10^5 - 10^2 \approxeq 10^5$$

$$ 10^5 - 10^2 = 99900 = 10^{\log_{10}(99900)} \approxeq 10^{4.999}  $$

$$ \log_{10}10^Q = Q $$

Say you have a logarthimic scale.

Adding logarithms is equivalent to adding linear distances on the chart, and the values inside the logarithms (the distances in space) will get multiplied.

If we a distance of $\log{10}(s) $ and then move an additional $\log_{10}(t) $ we have moved $\log_{10}st $

$$ \log_{10}s + log_{10}t = \log_{10}st $$

This is how much we have moved on the chart, not through space, so you're getting the point on the axis.

$$ \log_{10}10^2  + \log_{10}10^2 + \log_{10}10^2 = \log_{10}((10^2)(10^2)(10^2)) = \log_{10}(10^6) $$

If we move a distance of $\log_{10}(m) $ and repeat is $p$ times, so $p \times \log_{10}(m)$ we get $log_{10}(m^p)$ 

In the example, $p = 3$, so we have $(10^2)^3 = (10^6)$, not $(10^2 \times 3)$ which is $(m \times p)$

Take the inverse

$$ \log_{10}n - \log_{10}d = \log_{10}\frac{n}{d}$$

![]({{ site.baseurl }}/images/pre_calculus/log_arthimetic1.png){:width="300"}

$$ \log_b(s) + \log_b(t) = \log_b(st) $$
$$ \log_b(n) - \log_b(d) = \log_b(\frac{n}{d}) $$
$$ (p)\log_b(m) = \log_b(m^p) $$

$$ \log _{3}(x)+\log _{3}(y)-\log _{3}(z) = \log _{3}\left(\frac{x y}{z}\right)$$

Say we have;

$$y = \log{b}(ax) + c$$

![]({{ site.baseurl }}/images/pre_calculus/log_arthimetic2.png){:width="300"}

### Practice

(1) Express the following in a single log

$$ \log _{2}(x)+\log _{2}(x)+\log _{2}(x)+\log _{2}(x) $$

$$\log _{2}(x \cdot x \cdot x \cdot x)=\log _{2}\left(x^{4}\right) $$

(2) Solve for $x$ :
$$ \log _{8}(x)-\log _{8}(4)=\log _{8}(36)-\log _{8}(x) $$

$$
\begin{aligned}
\log _{8}(x)-\log _{8}(4) &=\log _{8}(36)-\log _{8}(x) \\
\log _{8}\left(\frac{x}{4}\right) &=\log _{8}\left(\frac{36}{x}\right) \\
\left(\frac{x}{4}\right) &=\left(\frac{36}{x}\right) \\
x^{2} &=144 \\
x &=12
\end{aligned}
$$

(3) Note 

$$\log_{b}(m) = y \equiv b^y = m $£

We also now that 

$$\log _{z}\left(b^{y}\right)=\log _{z}(m) $$

(3) If you now isolate $y$, what does it equal?

Applying the rule $(p) \log _{b}(m)=\log _{b}\left(m^{p}\right)$
$$ 
\begin{aligned}
\log _{z}\left(b^{y}\right) &=\log _{z}(m) \\
(y) \log _{z}(b) &=\log _{z}(m) \\
y &=\frac{\log _{z}(m)}{\log _{z}(b)}
\end{aligned} 
$$

****

## Change of base

> Change of base is a very useful procedure to manipulate log functions into something more useful.

$$
\log _{a}(b)=\frac{\log _{c}(b)}{\log _{c}(a)}
$$

$\text { If } \log _{9}(243)=2.5, \text { what is the value of } \log _{3}(243) ?$

$$
\log _{9}(243)=\frac{\log _{3}(243)}{\log _{3}(9)} = 2.5 \\
\frac{\log _{3}(243)}{2} = 2.5 \\
\log _{3}(243)= 5 \\
$$

Simplify;

$$ \frac{\log _{2}(a)}{\log _{4}(a)} $$

Instincitvely you know to are going to have to raise 4 by a power $x$ fewer times to reach $a$ than 2. That means $\log _{2}(a) > \log _{4}(a)$

We can apply the change of base formula to both the numerator and denominator. The new base we choose doesn't matter, so we will just use 10:
$$
\log _{2}(a)=\frac{\log _{10}(a)}{\log _{10}(2)}, \quad \log _{4}(a)=\frac{\log _{10}(a)}{\log _{10}(4)}
$$
Substituting back into our starting expression, we now have
$$
\frac{\frac{\log _{10}(a)}{\log _{10}(2)}}{\frac{\log _{10}(a)}{\log _{10}(4)}}=\frac{\log _{10}(4)}{\log _{10}(2)}
$$
Now $4=2^{2}$, so $\log _{10}(4)=\log _{10}\left(2^{2}\right)=2 \log _{10}(2)$, so we can rewrite the expression above as
$$
\frac{2 \log _{10}(2)}{\log _{10}(2)}=2
$$

****

## Logarithmic equations

{% include info.html text="When solving equations with logarithms, a general strategy is to rewrite the equation in exponential form" %}

$$\log _{a}(x)=y \text { is equivalent to } a^{y}=x $$

Solve for $x$: 

(1)
$$\log _{5}(x+1)=2 $$

$$ 5^2 = x+ 1 = 25$$

(2)
$$\log _{10}(3x+1)=2 $$

$$ 10^2 = 3x+ 1 = 100$$

(3)
$$\log _{4}(x) +log_{4}(x+6)=2 $$

$$ \log_{4}(x^2+6x)=2 $$

$$ 4^2=x^2 + 6x$$

$$ x^2 + 6x - 16 = 0$$

Using this along, we could consider $x$ to equal 2 or -8

$$ (x+8)(x-2)=0$$

However $\log _{4}(x) \text{ can't be defined for } x=-8$

(4)

$$
\left(\log _{4}(x)\right)^{2}+\log _{4}\left(x^{3}\right)-4=0
$$

Retwrite $log _{4}(x^{3})$ as $3\log_{4}(x)$

Substitute $\log_{4}(x)$ for $y$

$$ y^2 + 3y - 4 = 0 $$

$$ (y + 4 )(y - 1) = 0$$

$$ log_{4}(x) = -4 \text{ or } 1 $$

$$x=4^{-4}=\frac{1}{256} \quad \text { or } \quad x=4^{1}=4$$

(5)

$$\log _{2}(x)+\log _{4}(9)=\log _{2}(12)$$

Change base $\log _{4}(9) = \frac{log_2(9)} {log_2(4)}$

$$ log_2(x) + 1.5849625007211563 = 3.5849625007211565 $$

$$ log_2(x) = 2 $$

Note, could have rewritten the rebasing as:

$$
\frac{log_2(9)} {log_2(4)} = \frac{\log _{2}(9)}{2}=\left(\frac{1}{2}\right)\left(\log _{2}(9)\right)=\log _{2}\left(9^{\frac{1}{2}}\right)=\log _{2}(3)
$$
