import asyncio

async def greet():
    return "hello"

coro = greet() # nothig runs yet
print(type(coro)) # <class 'coroutine'>
print(asyncio.run(coro)) # hello (the event loop runs it)