# CSE573

This exercise is quite straightforward. We need to apply the Bellman equation in the two different situations. Let’s assume first that we want to go UP. We have:

\[U(s) = 0 + \gamma (\max_{a}\sum\limits_{s_{13}} P(s_{13}|s_{12},a) U(s_{13})) \\\]
If the agent goes UP we can only reach the state s13 with probability 1, so

\[\sum\limits_{s_{13}} P(s_{13}|s_{12},a) = 1 \times U(s_{13})\]
And finally we have:

\[U_{up}(s) = \gamma U(s_{13}) = \gamma(50 + \gamma \sum\limits_{s_{23}} P(s_{23} | s_{12},a) U(s_{23})) \\ = \gamma (50 + \gamma U(s_{23})) = 50 \gamma + \gamma^{2} U(s_{23}) \\ = 50 \gamma + \gamma^{2} (-1 + \gamma U(s_{33})) \\ = 50 \gamma - \gamma^{2} + \gamma^{3} (-1 + \gamma U(s_{43})) \\ = 50 \gamma - \gamma^{2} - \gamma^{3} + \gamma^{4} U(s_{43}) \\ = \text{...} \\ = 50 \gamma - \sum\limits_{i = 2}^{101} \gamma^{i} \\ = 50 \gamma - \gamma^{2} \sum\limits_{i=0}^{99} \gamma^{i} \\ = 50 \gamma - \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma}\]
The last relation comes from the fact that γ∈[0,1].

We use the Bellman equation to compute the utility if the agent goes DOWN. We obtain: \(U_{down}(s) = -50 \gamma + \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma}\)

We then need to solve the system (with a computer):

\[50 \gamma - \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma} = -50 \gamma + \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma}\]
to find the value of γ.
