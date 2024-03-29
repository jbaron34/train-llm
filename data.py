import json
import requests
from datetime import datetime
from enum import Enum

JOBS_API_URL = "http://20.213.17.116:8083/jobs/api/"
JOBS_SINCE = datetime(year=2024, month=1, day=1)
JOB_STATUS = "processed"
JOBS_JSONL_FILE = "jobs.jsonl"
MAX_NUM_JOBS = 10000


def download_jobs():
    first = True
    cursor = None
    params = {
        "updated": JOBS_SINCE,
        "status": JOB_STATUS,
    }

    while cursor or first:
        response = requests.get(JOBS_API_URL, params=params).json()
        jobs, cursor = response["jobs"], response["meta"].get("cursor")

        with open(JOBS_JSONL_FILE, "w" if first else "a") as f:
            f.writelines([json.dumps(job) + "\n" for job in jobs])

        params["cursor"] = cursor
        first = False


def filter_duplicate_jobs():
    with open(JOBS_JSONL_FILE, "r") as f:
        jobs = [json.loads(job) for job in f.readlines()]

    print(len(jobs))
    print(len([job for job in jobs if job["parameters"].get("expanded_prompt")]))

    most_recent_jobs = {}
    for job in jobs:
        if job["roomId"] not in most_recent_jobs or job["id"] > most_recent_jobs[job["roomId"]]["id"]:
            most_recent_jobs[job["roomId"]] = job
    jobs = list(most_recent_jobs.values())

    # print(json.dumps(next(iter(jobs.values())), indent=4, ensure_ascii=False))
    print(sum(len(job["parameters"].get("previous", [])) + 1 for job in jobs) / len(jobs))
    print(len(jobs))
    print(len([job for job in jobs if job["parameters"].get("expanded_prompt")]))

    with open("training_data.jsonl", "w") as f:
        for job in jobs:
            if job["task"] not in ("imagine", "imagine-turbo"):
                continue
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an ai image generator that takes user requests and interprets and converts them to optimized stable diffusion prompts",
                    },
                ]
                for prev_job in job["parameters"].get("previous", []):
                    messages.append({"role": "user", "content": prev_job["prompt"]})
                    messages.append({"role": "assistant", "content": prev_job["processed_prompt"]})
                messages.append({"role": "user", "content": job["parameters"]["prompt"]})
                messages.append({"role": "assistant", "content": job["parameters"]["processed_prompt"]})

                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            except Exception:
                print(json.dumps(job, indent=4))
                raise


def load_data():
    from datasets import load_dataset

    dataset = load_dataset("text", data_files="training_data.jsonl")
    print(len(dataset["train"]))
    return dataset


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def apply_template():
    dataset = load_data()
    from transformers import AutoTokenizer

    special_tokens = ChatmlSpecialTokens
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        pad_token=special_tokens.pad_token.value,
        bos_token=special_tokens.bos_token.value,
        eos_token=special_tokens.eos_token.value,
        additional_special_tokens=special_tokens.list(),
        trust_remote_code=True,
    )
    tokenizer.chat_template = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"

    conversation = [{"role": "system", "content": "You are an ai image generator that takes user requests and interprets and converts them to optimized stable diffusion prompts"}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy and unkempt and hes a white guy"}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He wears bulky clothes, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace and has unkempt brown hair."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary  black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy and unkempt and hes a white guy"}, {"role": "assistant", "content": "The image depicts a distinctive Caucasian man with a robotic eye, concealed by an eye patch. He is dressed in bulky clothes, including black sweatpants, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace. His unkempt brown hair adds to his intriguing persona."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy and unkempt and hes a white guy"}, {"role": "assistant", "content": "This digital art image portrays a unique white male with a robotic eye, concealed by an eye patch. He wears oversized clothing, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. A necklace of engraved metals and unkempt brown hair complete his look."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy and unkempt and hes a white guy"}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He is dressed in bulky clothes, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace and has unkempt brown hair."}, {"role": "user", "content": "In combat mode: He will wrap his arms and palms in rags showing barely little skin if they were poorly wrapped but most of the time hes very consistent with it. The multi metal chain would be tied around his waist band and pocket to not interfere with him in combat. His pants would be a mixtere of canvas and something similar to cotton to stay comfortable and to keep warm but they were tight to his skin and elastic for mobility. Their color was simply black and not flashy. His belt remained the same apart from a few canvas pouches being added to the left of his belt and the back and a scaley purple leather holster on his right which contained his main fire arm. His shirt was simply a white tank top with a small bullet resistance vest that looked useless but had a buffalo head painted on the front in rudimentary detail. He wore no gloves or jewelry apart from the chain in his waist band and pocket. He has a canvas duffle bag that is worn sideways to contain his ammunitions"}, {"role": "assistant", "content": "The image portrays a unique white male in combat mode. His arms are wrapped in rags and a multi-metal chain is tied around his waist. He wears tight, black, elastic pants, a white tank top, and a bullet-resistant vest with a buffalo head painted on it. His belt holds canvas pouches and a purple leather holster containing a firearm. He carries a canvas duffle bag for ammunition."}, {"role": "user", "content": "In combat mode: He will wrap his arms and palms in rags showing barely little skin if they were poorly wrapped but most of the time hes very consistent with it. The multi metal chain would be tied around his waist band and pocket to not interfere with him in combat. His pants would be a mixtere of canvas and something similar to cotton to stay comfortable and to keep warm but they were tight to his skin and elastic for mobility. Their color was simply black and not flashy. His belt remained the same apart from a few canvas pouches being added to the left of his belt and the back and a scaley purple leather holster on his right which contained his main fire arm. His shirt was simply a white tank top with a small bullet resistance vest that looked useless but had a buffalo head painted on the front in rudimentary detail. He wore no gloves or jewelry apart from the chain in his waist band and pocket. He has a canvas duffle bag His hair is straight narrow unkempt and dingy. He was sickly"}, {"role": "assistant", "content": "The image shows a unique white male in combat mode. His arms are wrapped in rags and a multi-metal chain is tied around his waist. He wears tight, black, elastic pants, a white tank top, and a bullet-resistant vest with a buffalo head painted on it. His belt holds canvas pouches and a purple leather holster containing a firearm. He carries a canvas duffle bag for ammunition."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are black tactical pants with a black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a white tank top with a tactical vest with a buffalo head painted on the front . He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy and unkempt. His robot eye was the only eye that wasn't covered by his eye patch. He has a side arm in his holster and pouches of ammunition in his belt pouched and a pack of smokes in his belt pouches."}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He wears tactical attire, including black pants with a buffalo-head buckle, a white tank top with a tactical vest, and black boots. He also wears a unique metal necklace, carries ammunition and smokes in his belt pouches, and has a sidearm in his holster."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy and unkempt and hes a white guy"}, {"role": "assistant", "content": "The image depicts a distinctive Caucasian man with a robotic eye, concealed by an eye patch. He is dressed in bulky clothes, including black sweatpants, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace. His unkempt brown hair adds to his intriguing persona."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy, unkempt and long. and hes a white guy"}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He wears bulky clothes, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace and has long, unkempt brown hair."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy, unkempt and long. and hes a white guy his facial features look sickly and seems to injured on his forearms"}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He wears bulky clothes, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace, has long unkempt brown hair, and exhibits sickly facial features with injuries on his forearms."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy, unkempt and long. and hes a white guy his facial features look sickly and seems to injured on his forearms"}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He wears bulky clothes, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace and has long, unkempt brown hair. His facial features appear sickly, and he has injuries on his forearms."}, {"role": "user", "content": "He tends to wear a eye patch over one of his eyes usually the robotic eye. His pupil in his normal eye is a dark magenta and his robotic eye is simply a round red dot in the center of a sphere in his eyesocket. His clothes are usually relatively bulky so his figure is never really certain as he tends to try and hide his physique and his flesh apart from his neck and head. His pants are usually black and baggy sweat pants with a unnecessary black scaley belt that has a buckle which emblems resembles a buffalo's head. His shoes are black leathery boots. He dawns a button up shirt that would seem like a dads hawaiian shirt but padded out to seem bigger and a long sleeved shirt underneath to hide his skin. He would wear a necklace of strange metals that seemed to have engravings on each different strain of metal on the chains. His hair was straight and brown looking dingy, unkempt and long. and hes a white guy his facial features look sickly and seems to injured on his forearms"}, {"role": "assistant", "content": "The image is a detailed digital art depiction of a white male with a robotic eye, concealed by an eye patch. He wears bulky clothes, including black sweatpants with a buffalo-head buckle, a padded Hawaiian shirt, and black boots. He also wears a unique metal necklace and has long, unkempt brown hair. His facial features appear sickly, and he has injuries on his forearms."}]
    encodeds = tokenizer.apply_chat_template(conversation, tokenize=False)

    print(encodeds)


if __name__:
    # download_jobs()
    # filter_duplicate_jobs()
    load_data()
    # apply_template()
