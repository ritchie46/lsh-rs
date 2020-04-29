LSH
===

Locality Sensitive Hashing can help you search through enormous data sets for approximated nearest neighbors.
If you want to read more about this algorithm try the following sources:

- `Introduction <http://people.csail.mit.edu/gregory/annbook/introduction.pdf>`_

The gist of the algorithm is that data points (vectors) that are close in some high dimensional space will be likely to
have the same hash. The hash functions we choose to hash the vectors determine the distance function we use to define
"closeness". At the moment we expose the following hashers:

+-------------------------+----------------------+
| Hasher                  | Distance/ similarity |
+=========================+======================+
| Sign Random Projections | Cosine similarity    |
+-------------------------+----------------------+
| P-stable distributions  | Euclidean            |
+-------------------------+----------------------+

Hyperparameters
---------------
The LSH algorithm requires two hyperparameters:

* The length of the generated hash **k**. A larger value for **k** leads to less hash collisions, thus faster query times.
* The number of different hash tables **L**. There will be **L** hash tables with **L** randomly generated hash functions.

The **L** hyperparamter can be derived from the query success probability and **k**. Read my
`blog post <https://www.ritchievink.com/blog/2020/04/07/sparse-neural-networks-and-hash-tables-with-locality-sensitive-hashing/#2%20Locality%20Sensitive%20Hashing>`_
on that subject to get an explanation.


L2
..

The L2 LSH has an additional hyperparameter **r**. This is the width of bucket hash values can fall in.
If you normalize your data by the distance threshold :math:`R` this hyperparameter should be approximately 4.
