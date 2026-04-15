from router import TopicRouter

router = TopicRouter("router_artifacts/router_context.joblib")

topic, conf = router.predict(
    "Räägi veel sellest.",
    last_topic="kasitoo",
    turn_index=2,
    is_follow_up=1,
)

print(topic, conf)
