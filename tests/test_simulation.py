from eyetrack2llm.simulation import ResidualSimulationConfig, run_residual_recovery_simulation


def small_result(seed=17):
    return run_residual_recovery_simulation(ResidualSimulationConfig(
        subjects=(12,), latent_effects=(0.0, 0.65), concentrations=(30.0,), replicates=12,
        seed=seed, n_sources=18, events_per_subject_source=20,
    ))


def test_simulation_seed_is_deterministic():
    assert small_result(11) == small_result(11)


def test_correct_residual_is_centered_under_null():
    rows = small_result()["summary"]
    null = next(row for row in rows if row["method"] == "correct" and row["latent_effect"] == 0)
    assert abs(null["latent_recovery_correlation_mean"]) < 0.12


def test_positive_effect_recovers_more_than_null():
    rows = small_result()["summary"]
    correct = {row["latent_effect"]: row for row in rows if row["method"] == "correct"}
    assert correct[0.65]["latent_recovery_correlation_mean"] > correct[0.0]["latent_recovery_correlation_mean"] + 0.25
