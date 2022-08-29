



Godot:
PYTHONPATH=$PYTHONPATH:../../science:../../computronium python -m agent.train_dreamer_godot


Hyperparam opt:
PYTHONPATH=$PYTHONPATH:../../science:../../computronium:../../bones python opt_dreamer.py train
- train for a single run
- run for a sweep


## Benchmarks I've tested on

See `launch_tests.py` for more.

- state-based
  - custom test envs:
    - test1: good. disclaim = 1, .5
    - test6: good, with reinforce. good, at least sometimes, with dynamics.
  - gym-cartpole (discrete)
      - tests short episodes (without time limit)
      - reinforce:
          - hyperparams look kinda bad, but learns well
      - dynamics
          - looks like it would probs work with tuned hyperparams.
          - actually idk, policy is pretty unstable while the rest of the model looks solid.
  - gym Pendulum-v1 (continuous)
    - learns well and then decays badly. not sure what's up with that.
  - don't yet have a wrapper to run mujoco envs just with state. should add that.
  - want to add an env with aggressive time limits. probably one of the above with custom time limits applied.
- vision-based:
  - dmc suite. danijar published scores for v1 (they don't match his impl though) and v2 (haven't checked vs his impl)
- hybrid observation:
  - custom test env hybrid1
  - our godot env


### DMC Vision-based

Cartpole-balance:
- My v1 impl:
  - https://wandb.ai/sourceress/zack_dreamer_new/runs/2sfylt5x?workspace=user-zplizzi
- Danijar's v2 impl, but with some custom hyperparams:
  - ie continuous latents
  - https://wandb.ai/sourceress/zack_dreamer_new/runs/54dbfgit?workspace=user-zplizzi



## How this implementation compares to dreamerv1/v2

- start with dreamerv1
- add from dreamerv2:
    - remove exploration noise, replace with sampling + entropy loss
    - discrete action option
    - reinforce option
    - target network for value training
    - KL balancing
- removed the sampling bug/fix from dreamerv1
- switched from v2 entropy loss to an uninformative KL loss
  - only really affects plain Normal distributions
- two methods for clipped Normal distributions:
  - a plain Normal distribution for training, and clip actions in a wrapper
  - a SampleDist for a tanh-transformed normal dist
  - (haven't implemented the TruncatedNormal dist from dreamerv2
    - tried implementing this but the sampling impl isn't numerically stable for means much outside of the bounds, so it explodes quickly. Could try a wide tanh into this dist, like we do with the tanh-transformed dist.
- hybrid dynamics/reinforce loss
- major things still not implemented from v2:
    - discrete latent
    - LOTS of smaller implementation stuff

(and probably stuff not listed here)


Little things to change to get closer to dreamerv2:
- pretrain agent
- weight decay
- train_every = 5


## Thoughts/concerns

- if pcont drops close to 0 at any point, discount will be near-zero following that. So errors in the pcont model are problematic as they create high variance in the value target.
- there doesn't seem to be any method for ensuring the entropy of the latent distributions (prior and posterior stoch portion) doesn't go to 0 (or below..?), since the loss just ensures they are close. In a normal VAE we draw these towards a Normal(0, 1). Should that be done here also? In practice I see the entropy generally decrease thru the runs and only occasionally increase.



## Ideas for improvement

There's probably still room to improve by pulling in more stuff from dreamerv2. Eg this cartpole run on danijar's code is way better than what my impl does: https://wandb.ai/sourceress/zack_dreamer_new/runs/54dbfgit?workspace=user-zplizzi

- look thru todos in code!
- the way the observation is reconstructed seems interesting to me.
  - the reconstruction target is "next_obs", from the tuple (action, reward, next_obs).
  - which i guess makes sense, given that the state is (obseration, action). next_obs is more representative of that state than prev_obs, at least in deterministic envs.
  - but in terms of predictions, being able to reconstruct prev_obs is more useful for predicting eg the reward and done signal.
  - and also, i think reconstructing next_obs will bypass part of the dynamics network - since grads aren't flowing thru a dynamics step. since obs is our highest-data training signal, having grads flow thru the dynamics is maybe helpful for training that.
  - so i'd be curious to try having both next_obs and prev_obs as targets.
- the KL div loss between prior and posterior is obviously critical. but it's not perfect:
  - in the dreamerv1 impl, we just take the div and give it some free nats. this is not great because the training is jumpy - mostly no signal when div is under this bound, and then a sharp signal when it drifts out. and we can't bring them fully together in the limit.
  - in dreamerv2, this is replaced with kl balancing with no free nats in at least some of the hyperparam configs. this slowed down training of the model in some envs because we'd get "stuck" in a situation where the KLs were closely aligned before learning anything, and learning to actually use the posterior didn't happen because that would harm the KL loss more than it would help, at least initially. so eg the reward loss was very flat, learning very slowly, and then would quickly jump to a new value once the learning invisibly progressed far enough to overpower the KL loss.
    - but we can't just drop the KL penalty, because the penalty needs to be large enough to prevent "cheating". if i'm understanding cheating properly. maybe i'm not and there's no such thing as cheating. but if the penalty is too low, since part of the dynamics is shared, we might not get enough weight to the gradients to learn what's useful for the prior.
  - there may be a better way to balance these tradeoffs

## How Dreamer works

TODO:
- is my understanding of "cheating" accurate now that i'm realizing that next_obs is a pretty good representation of the state? it's still seeing something that hasn't been observed, but in a deterministic env nothing that couldn't be predicted.

- Collect sequences of (action -> step -> (reward, done, next_obs)) tuples.
  - Note this is different to models i'm used to, which use tuples of (obs -> computed action+policy -> (step taken) -> reward). The difference being a shift of the observations by 1.
- The key thing to understand is: a "state" is predictive of the reward and done for the *same* timestep, ie before we take any more actions. The "state" is the state of the world when we have observed the environment, decided on an action, but not yet observed the results. In effect, it's more like a Q(s, a) than a V(s), if that makes sense. This differs from my models, where a state would be used to predict an action, which would be used to take a step and get a reward. So the state did not contain the already-decided-upon action.
  - to be more specific:
    - in my models, R(s) is the reward received after selection A(s) and taking it. In their models, R(s) is the reward received after taking the action that led to state s.
    - in my models, V(s) is the value of the state before deciding on an action. In dreamer, V(s) is more like a Q(s, a) - it's the value knowing the action that led to state s.
  - To be more intuitive:
    - In my models, a "state" is what you get when you see an observation. You look at the screen, you see where the ball and paddles are, and you have a "state". It's purely an observation.
    - In dreamer, a "state" is the state of the world when you see the observation and decide on what action to take, but you haven't yet taken it. It's an (observation, action) pair.
  - Interestingly, this isn't just a curious implementation detail. It affects the difficulty of the various prediction problems.
    - Predicting a policy from a (s, a) pair is harder - we haven't "stepped" the env after the last action, so the policy model kinda has to do that on its own. Or idk maybe we have - when to internally compute that is a choice the model gets to make i guess?
    - Predicting a reward is different - it's no longer conditional on the policy, but is just dependent on the environment since the action is specified.
    - Predicting a value is.. idk? 
- The dynamics model:
  - has a prior and a posterior.
    - prior: new_state = f(last_state, action_computed_from_last_state)
      - this is the usual dynamics model.
    - posterior: new_state = f(last_state, action_computed_from_last_state, next_obs)
      - this is how we encode observations into states.
      - in my models, i would directly have an encoder: state = f(current_obs)
      - i think this formulation is inspired by something more like a kalman filter. still don't quite understand the tradeoffs though.
    - in practice, the prior and posterior share some components:
      - they both share a model: deter = f(last_state, action)
          - f() is a single application of a GRU cell, where the "state" is the deter part of last_state, and the "input" is the stoch part of last_state concated with the action
          - so over multiple step rollouts, there's a direct sequence of deter->GRU->deter->GRU. should allow for clean gradients. and the "stoch" can be seen as heads off of this main state (but the heads also feed back into the GRU input).
      - the prior then adds a stochastic head on this with no additional input.
      - the posterior concats an encoding of the observation, and adds a stochastic head.
      - in both cases, the resulting "state"/"latent" vector is a concatenation of `deter` plus a sample from `stoch`. so the `deter` part of the latent will be identical in both.
    - what does "state" actually represent?
- Training:
  - world model:
    - we train by feeding trajectories through the "observation" step, which simply sequentially applies the posterior model.
      - the prior model is trained by a KL loss between prior and posterior at each step.
        - primarily we want to train the stoch head of the prior model here.
        - **but this will also train the posterior not to include things that aren't predictable.**
          - this is actually critical, i think, to forcing the state to not "cheat" and include too much stuff from `next_obs` which could never be predicted.
      - the heads are trained by an autoencoding loss between the input and decoded prediction at each step, where the decoding is done with the posterior latent.
        - so gradients flow from the prediction error, thru the decoder, back into the posterior computation (including the part shared with the prior), and so thru the entire sequential application of the posterior model, and the encoding of inputs.
        - notably, the heads are not trained at all on just the priors. i'm surprised by this. in my models, i would use multi-step imagination rollouts (with ground truth actions) and compare to ground truth.
      - the reward is trained just like any other head. notably though, predicting reward is probably often very easy given the following observation. so the posterior state probably is very good at predicting reward, and that doesn't necessarily mean that the prior is similarly good.
  - value:
    - trained only on imaginary rollouts with the prior, because the stored rollouts would be off-policy. there's not a readily obvious way to make dynamics backprop dreamer work on off-policy rollouts (but wouldn't be too hard with REINFORCE, just switch to PPO).
  - when training with REINFORCE - what is unique about this vs other works? just the specific implementation of the world model? because dynamics backprop is really the unique contribution of dreamer.
- "observation":
  - at each step, we pass in the previous state, and this timestep's action and observation. the action preceeds the observation.


- i guess i just don't have a good intuitive grasp of how the posterior model is incorporating the observation.
  - it gets worked into the posterior's `stoch` component of the latent.
  - i guess the weird thing is that there's conflicting goals.
      - there's the goal to help future predictions. in future preds, having seen this won't be "cheating", as this observation will be contemporaneous or earlier than the targets. so we just want to store as much relevant info as possible to use in the future.
          - specifically, if we store all the info, we can perfectly predict next timestep's obs. if we do this, the training observation loss will be 0.
          - but this info isn't necessarily relevant to the current step. eg storing "the ball turns blue next step!" isn't really part of the current "state". so it feels weird to me to have this stuff in the state vector.
      - then there's the goal of storing info relevant to predicting the last step's observation.
          - which is kind of weird - to do this well, we actually need to be essentially doing dynamics in reverse!
          - and if it's "cheat" info, then we're gonna penalize usage of it.
      - perhaps in practice, we do whatever works initially, until the prior learns how the dynamics work. once that happens, the most "loss-efficient" approach is def the latter option, since we're not wasting unpredictable bits just to aid future predictions.
      - still feels weird tho, **what if we gave the posterior both the current and future state? then we wouldn't have to also be learning reverse dynamics**.

How Dreamer differs to my implementation of model-based RL:
- see above how the concept of a "state" is different - containing or not containing an action. (s, a) vs (s)
- my impl:
    - I had a single dynamics model, new_state = f(last_state, action)
    - and an encoding model: state = f(obs)
    - and i trained my dynamics model by autoencoding with multiple applications of the dynamics model in the middle (autoencoding thru time).
    - i also tried training the dynamics by running it for n steps and comparing to the encoding of the ground truth n steps into the future.
- their impl (described above):
    - they don't ever directly train the prior model (equivalent to my dynamics model).
    - instead they train a posterior, which is a prior augmented with "cheat" info by allowing it to see the future. so it should be a "stronger" version of the prior.
        - it's constrained to not cheat too much by not allowing it to stray far from the prior
    - and the prior is trained to match the posterior
    - the posterior is trained by sort-of-autoencoding.
      - we don't directly give the posterior this timestep's observation, yet we request this timestep's decoding - so it's not quite a proper autoencoding.
      - so eg, autoencoding of the observation will result in training the latent by either pulling info from next_obs (somewhat cheating, but we want that to some extent) or pulling info from the previous applications of the posterior model which had access to the previous observations. so sort of, we are autoencoding through time.
- **why their approach is more elegant than mine**:
  - mine doesn't allow any way to combine observations from multiple timesteps into a single latent. thus it's limited to environments that can be made fully-observable-enough just by framestacking or similar.
  - theirs can cleanly handle partially observable environments.
- implementation-level differences:
  - in my model, f() was convolutional i think?
  - in their model, f() is a single application of a GRU cell, where the "state" is the deter part of the last state, and the "input" is the stoch part of the last state concated with the action
