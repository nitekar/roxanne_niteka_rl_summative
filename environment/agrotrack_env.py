import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any

class AgroTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=10, max_inventory_items=8, max_steps=100, render_mode: Optional[str]=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_inventory_items = max_inventory_items
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(9)
        obs_size = 2 + 2 + 3 + (max_inventory_items*5) + 4
        self.observation_space = spaces.Box(low=-1.0, high=10.0, shape=(obs_size,), dtype=np.float32)

        self.crop_types = ['cassava','maize','beans','banana','rice','potato']
        self.agent_pos = np.array([0,0])
        self.inventory = []
        self.temperature = 25.0
        self.humidity = 60.0
        self.preservation_resource = 100.0
        self.transport_resource = 100.0
        self.storage_capacity = 100.0
        self.current_step = 0
        self.total_loss = 0.0
        self.total_saved = 0.0

        self.window = None
        self.clock = None
        self.cell_size = 60

    def _get_obs(self) -> np.ndarray:
        obs = []
        obs.extend(self.agent_pos/self.grid_size)
        obs.append(self.temperature/40.0)
        obs.append(self.humidity/100.0)
        obs.append(self.preservation_resource/100.0)
        obs.append(self.transport_resource/100.0)
        obs.append(self.storage_capacity/100.0)

        for i in range(self.max_inventory_items):
            if i<len(self.inventory):
                item=self.inventory[i]
                obs.extend([item['x']/self.grid_size, item['y']/self.grid_size,
                            item['freshness']/100.0, item['quantity']/100.0,
                            item['spoilage_risk']])
            else:
                obs.extend([0]*5)

        demand=np.random.uniform(0.5,1.0,4)
        obs.extend(demand)
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {'total_loss':self.total_loss,'total_saved':self.total_saved,
                'avg_freshness': np.mean([i['freshness'] for i in self.inventory]) if self.inventory else 0,
                'items_at_risk': sum(1 for i in self.inventory if i['freshness']<30),
                'preservation_resource':self.preservation_resource,
                'transport_resource':self.transport_resource}

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self.agent_pos=np.array([self.grid_size//2, self.grid_size//2])
        self.current_step=0
        self.total_loss=0
        self.total_saved=0
        self.preservation_resource=100.0
        self.transport_resource=100.0
        self.storage_capacity=100.0
        self.temperature=np.random.uniform(20,32)
        self.humidity=np.random.uniform(50,80)
        self.inventory=[]
        n_items=np.random.randint(4,self.max_inventory_items+1)
        for i in range(n_items):
            self.inventory.append({'id':i,
                                   'x':np.random.randint(0,self.grid_size),
                                   'y':np.random.randint(0,self.grid_size),
                                   'crop_type':np.random.choice(self.crop_types),
                                   'freshness':np.random.uniform(60,100),
                                   'quantity':np.random.uniform(10,100),
                                   'spoilage_risk':np.random.uniform(0.1,0.6)})
        obs=self._get_obs()
        info=self._get_info()
        if self.render_mode=="human":
            self._render_frame()
        return obs, info

    def step(self, action:int):
        self.current_step+=1
        reward=0.0
        # Movement
        if action==0: self.agent_pos[1]=max(0,self.agent_pos[1]-1)
        elif action==1: self.agent_pos[1]=min(self.grid_size-1,self.agent_pos[1]+1)
        elif action==2: self.agent_pos[0]=max(0,self.agent_pos[0]-1)
        elif action==3: self.agent_pos[0]=min(self.grid_size-1,self.agent_pos[0]+1)
        items_here=[i for i in self.inventory if i['x']==self.agent_pos[0] and i['y']==self.agent_pos[1]]
        # Task actions
        if action==4 and items_here: reward+=1
        elif action==5 and items_here and self.preservation_resource>=10:
            for i in items_here:
                i['freshness']=min(100,i['freshness']+np.random.uniform(5,15))
                reward+=5
                self.total_saved+=i['quantity']*0.1
            self.preservation_resource-=10
        elif action==6 and items_here:
            for i in items_here:
                if i['freshness']<50:
                    reward+=10
                    self.total_saved+=i['quantity']*0.5
                    self.inventory.remove(i)
                else: reward-=2
        elif action==7 and items_here and self.transport_resource>=15:
            for i in items_here:
                i['freshness']=min(100,i['freshness']+8)
                i['spoilage_risk']*=0.7
                reward+=7
            self.transport_resource-=15
        elif action==8: reward-=0.5

        # Environmental decay
        temp_factor=max(0,(self.temperature-25)/10)
        hum_factor=max(0,(self.humidity-60)/20)
        to_remove=[]
        for i in self.inventory:
            decay=i['spoilage_risk']*(1+temp_factor+hum_factor)
            i['freshness']-=decay
            if i['freshness']<=0:
                to_remove.append(i)
                self.total_loss+=i['quantity']
                reward-=15
        for i in to_remove: self.inventory.remove(i)

        self.temperature=np.clip(self.temperature+np.random.uniform(-1,1),18,35)
        self.humidity=np.clip(self.humidity+np.random.uniform(-2,2),40,90)
        self.preservation_resource=min(100,self.preservation_resource+0.5)
        self.transport_resource=min(100,self.transport_resource+0.3)

        if self.inventory:
            avg_freshness=np.mean([i['freshness'] for i in self.inventory])
            reward+=2 if avg_freshness>70 else -3 if avg_freshness<40 else 0

        terminated=len(self.inventory)==0
        truncated=self.current_step>=self.max_steps
        obs=self._get_obs()
        info=self._get_info()
        if self.render_mode=="human": self._render_frame()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode=="rgb_array": return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode=="human":
            pygame.init()
            pygame.display.init()
            self.window=pygame.display.set_mode((self.grid_size*self.cell_size,self.grid_size*self.cell_size+100))
            pygame.display.set_caption("AgroTrack RL")
        if self.clock is None and self.render_mode=="human": self.clock=pygame.time.Clock()

        canvas=pygame.Surface((self.grid_size*self.cell_size,self.grid_size*self.cell_size+100))
        canvas.fill((245,245,240))
        # Grid
        for x in range(self.grid_size+1):
            pygame.draw.line(canvas,(200,200,200),(x*self.cell_size,0),(x*self.cell_size,self.grid_size*self.cell_size),1)
        for y in range(self.grid_size+1):
            pygame.draw.line(canvas,(200,200,200),(0,y*self.cell_size),(self.grid_size*self.cell_size,y*self.cell_size),1)
        # Items
        for i in self.inventory:
            color=(74,222,128) if i['freshness']>70 else (250,204,21) if i['freshness']>40 else (239,68,68)
            pygame.draw.rect(canvas,color,(i['x']*self.cell_size+5,i['y']*self.cell_size+5,self.cell_size-10,self.cell_size-10))
            font=pygame.font.Font(None,20)
            canvas.blit(font.render(f"{int(i['freshness'])}%",True,(0,0,0)),(i['x']*self.cell_size+12,i['y']*self.cell_size+25))
        # Agent
        pygame.draw.circle(canvas,(59,130,246),(self.agent_pos[0]*self.cell_size+self.cell_size//2,self.agent_pos[1]*self.cell_size+self.cell_size//2),self.cell_size//3)
        # Info
        info_y=self.grid_size*self.cell_size+10
        font=pygame.font.Font(None,24)
        x_offset=10
        for text in [f"Step:{self.current_step}/{self.max_steps}",f"Items:{len(self.inventory)}",
                     f"Loss:{self.total_loss:.1f}",f"Saved:{self.total_saved:.1f}",
                     f"Temp:{self.temperature:.1f}°C",f"Humidity:{self.humidity:.1f}%"]:
            canvas.blit(font.render(text,True,(0,0,0)),(x_offset,info_y))
            x_offset+=120
        if self.render_mode=="human":
            self.window.blit(canvas,canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),axes=(1,0,2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
