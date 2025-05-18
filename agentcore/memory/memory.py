from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Union
import json
import logging
from aucodb.graph import LLMGraphTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMeta(ABC):
    @abstractmethod
    def update_memory(self, graph: list):
        pass
    
    @abstractmethod
    def save_short_term_memory(self, llm, message):
        pass

    @abstractmethod
    def save_memory(self, message: str, *args, **kwargs):
        pass

class Memory(MemoryMeta):
    '''This stores the memory of the conversation.
    '''
    def __init__(self, 
            memory_path: Optional[Union[Path, str]] = Path('templates/memory.jsonl'), 
            is_reset_memory: bool=False,
            is_logging: bool=False,
        *args, **kwargs):
        if isinstance(memory_path, str) and memory_path:
            self.memory_path = Path(memory_path)
        else:
            self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_reset_memory = is_reset_memory
        self.is_logging = is_logging
        if not self.memory_path.exists():
            self.memory_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
        if self.is_reset_memory:
            self.memory_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

    def load_memory(self, load_type: Literal['list', 'string'] = 'list', user_id: str = None):
        data = []
        with open(self.memory_path, "r", encoding="utf-8") as f:
                # for line in f:
                #     try:
                #         route = json.loads(line)
                #         if route:
                #             if user_id:
                #                 if route['user_id'] == user_id:
                #                     data.append(route)
                #             else:
                #                 data.append(route)
                #     except json.JSONDecodeError as e:
                #         logger.warning(
                #             f"Skipping invalid JSON line: {line.strip()} - Error: {e}"
                #         )

                data = json.load(f)
                if not user_id: # Load all memory
                    data_user = data
                else: 
                    if user_id in data: # Load memory by user_id
                        data_user = data[user_id]
                    else:
                        data_user = []

        if load_type == 'list':
            return data_user
        elif load_type == 'string':
            message = self.revert_object_mess(data_user)
            return message

    def save_memory(self, obj: list, memory_path: Path, user_id: str):
        memory = self.load_memory(load_type='list')
        memory[user_id] = obj
        with open(memory_path, "w", encoding="utf-8") as f:
            # for item in obj:
            #     f.write(json.dumps(item) + '\n')
            json.dump(memory, f, indent=4, ensure_ascii=False)

        if self.is_logging:
            logger.info(f"Saved memory!")

    def save_short_term_memory(self, llm, message, user_id):
        graph_transformer = LLMGraphTransformer(
            llm = llm
        )
        graph = graph_transformer.generate_graph(message)        
        self.update_memory(graph, user_id)
        return graph

    def revert_object_mess(self, object: list[dict]):
        mess = []
        for line in object:
            head, _, relation, relation_properties, tail, _ = list(line.values())
            relation_additional= f"[{relation_properties}]" if relation_properties else ""
            mess.append(f"{head} -> {relation}{relation_additional} -> {tail}")
        mess = "\n".join(mess)
        return mess

    def update_memory(self, graph: list, user_id: str):
        memory_about_user = self.load_memory(load_type='list', user_id=user_id)
        if memory_about_user:
            index_memory = [(item['head'], item['relation'], item['tail']) for item in memory_about_user]
            index_memory_head_relation_tail_type = [(item['head'], item['relation'],  item['tail_type']) for item in memory_about_user]
        else:
            index_memory = []
            index_memory_head_relation_tail_type = []
            
        if graph:
            for line in graph:
                head, head_type, relation, relation_properties, tail, tail_type= list(line.values())
                lookup_hrt = (head, relation, tail)
                lookup_hrttp = (head, relation, tail_type)
                if lookup_hrt in index_memory:
                    if self.is_logging:
                        logger.info(f"Bypass {line}")
                    pass
                elif lookup_hrttp in index_memory_head_relation_tail_type:
                    index_match = index_memory_head_relation_tail_type.index(lookup_hrttp)
                    if self.is_logging:
                        logger.info(f"Update new line: {line}\nfrom old line {memory_about_user[index_match]}")
                    memory_about_user[index_match] = line
                else:
                    if self.is_logging:
                        logger.info(f"Insert new line: {line}")
                    memory_about_user.append(line)
        else:
            if self.is_logging:
                logger.info(f"No thing updated")
        
        self.save_memory(obj=memory_about_user, memory_path=self.memory_path, user_id=user_id)
        return memory_about_user
