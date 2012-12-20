# TODO #
* numerical stability; how does sigma\_n become non-symmetric?
* AR pass getitem
* draw tracked trajectories! store with particle (track), include states, draw 5
  most probable
    - is it getting the state labeling correct? all else is gravy
* GPU implementation of naive momentum particles

# Thinking #
* Options to try
    1. smart particles
    2. naive particles + Gibbs
    3. naive particles at start, then Gibbs, then smart particles

