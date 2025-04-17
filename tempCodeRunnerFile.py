for i, path in enumerate(model_paths):
#     runs = 2 if i == len(model_paths) - 1 else 1  # Run last model twice

#     for run_idx in range(runs):
#         level = 1 if run_idx == 0 else 2  # Increment level for second run of last model
#         print(f"Running model: {path} | Level: {level}")

#         # Create env and model
#         env = ShooterEnv(render_mode="human", start_level=level)
#         obs_size = env.observation_space.shape[0]
#         n_actions = env.action_space.n
#         model = DQN(obs_size, n_actions)

#         # Load model
#         model.load_state_dict(torch.load(path, map_location=device))
#         model.to(device)

#         # Visualize
#         reward = visualize_episode(model, device)
#         print(f"Total reward: {reward}\n")

#         env.close()