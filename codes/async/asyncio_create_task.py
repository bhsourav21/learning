import asyncio
from datetime import datetime

def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')}] {message}")

async def step(name, seconds):
    log(f"{name} started")
    await asyncio.sleep(seconds)
    log(f"{name} finished")
    return name

async def main():
    t1 = asyncio.create_task(step("embed", 7))
    t2 = asyncio.create_task(step("search", 5))
    t3 = asyncio.create_task(step("embed2", 3))
    t4 = asyncio.create_task(step("search2", 1))
    # await t1
    await t2

asyncio.run(main())

# Fundamental
## create_task first creates all four tasks t1, t2, t3 and t4.
## So when all the asyncio.create_task commands have run, we have 4 tasks
## ready to be executed by not execute yet.
## Now await <task> runs.
## It starts all the tasks waiting, so all 4 tasks started
## So it always prints -
## [11:08:20.291952] embed started
## [11:08:20.291993] search started
## [11:08:20.292009] embed2 started
## [11:08:20.292018] search2 started
## Now the actual catch. whatever task is awaited, it waits for all the 
## tasks that executed prior to that task and including the task. Let me 
## explain.

## case 1:
## t1 = asyncio.create_task(step("embed", 7))
## t2 = asyncio.create_task(step("search", 5))
## t3 = asyncio.create_task(step("embed2", 3))
## t4 = asyncio.create_task(step("search2", 1))
## await t1
## Here await is execute for t1. That waits for 7 seconds. By the time
## t4, t3 and t2 have completed. So the output is - 
# [11:18:34.037332] embed started
# [11:18:34.037381] search started
# [11:18:34.037395] embed2 started
# [11:18:34.037404] search2 started
# [11:18:35.038707] search2 finished
# [11:18:37.038732] embed2 finished
# [11:18:39.038747] search finished
# [11:18:41.038644] embed finished

## case 2:
## t1 = asyncio.create_task(step("embed", 7))
## t2 = asyncio.create_task(step("search", 5))
## t3 = asyncio.create_task(step("embed2", 3))
## t4 = asyncio.create_task(step("search2", 1))
## await t2
## Here await is execute for t2. That waits for 5 seconds. By the time
## t4, t3 have completed. So the output is - 
# [11:18:34.037332] embed started
# [11:18:34.037381] search started
# [11:18:34.037395] embed2 started
# [11:18:34.037404] search2 started
# [11:18:35.038707] search2 finished
# [11:18:37.038732] embed2 finished
# [11:18:39.038747] search finished
## it does not wait for t1 which is executed after 7 seconds
