#just to check if teh text generator is working as expected and not rambling or echoing the question
import text_generator
text_generator.load_generator.cache_clear()
print(text_generator.load_generator().__dict__.get("task"))  # -> text2text-generation
print(
    text_generator.generate_text(
        "A message to a friend who failed an interview.",
        emotions=["caring","relief"],
        max_words=100,
    )
)
