import brax
from brax import envs
import jax
import jax.numpy as jnp
from gym import spaces

def test_done_signal(env_name, max_steps=1000):
    # 初始化 Brax 环境
    env = envs.create(env_name=env_name, batch_size=1)
    state = env.reset(rng=jax.random.PRNGKey(seed=0))
    env.step = jax.jit(env.step)
    for step in range(max_steps):
        # 随机生成动作
        action = jax.device_get(env.sys.actuator.ctrl_range)
        action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')
        action = jnp.array(action_space.sample())
        action = jnp.expand_dims(action, axis=0)
        
        # 环境步进
        state = env.step(state, action)

        # 检查是否 done
        if state.done.any():
            print(f"Step {step}: done signal detected.")
            
            # 检查是否是因为 truncation
            if 'truncation' in state.info and state.info['truncation']:
                print("Termination due to truncation (time limit reached).")
            else:
                print("Natural termination (task succeeded or failed).")
            
            # 可以选择在检测到 done 后重置环境或结束测试
            state = env.reset(rng=jax.random.PRNGKey(seed=step))
        else:
            # print(f"Step {step}: environment is still running.")
            pass
    
    print("Test completed.")

if __name__ == "__main__":
    # 指定你想测试的Brax环境名称
    test_done_signal('ant', max_steps=5000)
