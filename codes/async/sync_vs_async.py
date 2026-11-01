import asyncio, time

def call_model_sync(prompt):
    time.sleep(1) # pretend: waiting for AI model
    return f"answer to {prompt}"

async def call_model_async(prompt):
    await asyncio.sleep(1) # waiting, but lets other run
    return f"answer to {prompt}"

def run_sync():
    start = time.perf_counter()
    results = [call_model_sync(p) for p in ["A", "B", "C"]]
    print(f"sync: {time.perf_counter() - start:.1f}s", results)

async def run_async():
    start = time.perf_counter()
    coroutine_create = [call_model_async(p) for p in ["A", "B", "C"]]
    print(f"coroutine_create:{coroutine_create}")
    # results = await asyncio.gather(
    #     *(call_model_async(p) for p in ["A", "B", "C"])
    # )
    results = await asyncio.gather(*coroutine_create)
    print(f"async: {time.perf_counter() - start:.1f}s", results)

run_sync()
asyncio.run(run_async())