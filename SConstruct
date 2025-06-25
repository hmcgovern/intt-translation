import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
from glob import glob
import time
import sys
import json
from steamroller import Environment

# workaround needed to fix bug with SCons and the pickle module
# del sys.modules['pickle']
# sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


# actual variable and environment objects
vars = Variables()
# NOTE: uncomment if you have variables in custom.py to add, like GPU account
vars = Variables("custom.py")


vars.AddVariables(
    ("DATA_PATH", "", "data"),
    ("WORK_DIR", "", "work"),
    ("BIBLE_CORPUS", "", "${DATA_PATH}/bibles2"),
    ("CROSS_REFERENCE_FILE", "", "${DATA_PATH}/biblical-cross-references.txt"),
    ("STEP_BIBLE_PATH", "", "${DATA_PATH}/STEP"),
    ("DEVICE", "", "cuda"),
    ("BATCH_SIZE", "", 50),
    ("VOTE_THRESHOLD", "", 50),
    (
        "LANGUAGE_MAP",
        "",
        {
            "Hebrew" : ("he_IL", "heb_Hebr", "he", "Ancient_Hebrew"),
            "Greek" : ("el_XX", "ell_Grek", "el", "Ancient_Greek"),
            "English" : ("en_XX", "eng_Latn", "en", "English"),
            "Finnish" : ("fi_FI", "fin_Latn", "fi", "Finnish"),
            "Turkish" : ("tr_TR", "tur_Latn", "tr", "Turkish"),
            "Swedish" : ("sv_SE", "swe_Latn", "sv", "Swedish"),
            "Marathi" : ("mr_IN", "mar_Deva", "mr", "Marathi"),
        }
    ),
    (
        "ORIGINALS",
        "",
        [
            {"testament" : "OT", "language" : "Greek", "form" : "Septuagint", "file" : "${DATA_PATH}/LXX/LXX_aligned.json", "manuscript" : "LXX"},
            {"testament" : "NT", "language" : "Greek", "form" : "Septuagint", "file" : "${STEP_BIBLE_PATH}/NT_aligned.json", "manuscript" : "TAGNT"},
            {"testament" : "OT", "language" : "Hebrew", "form" : "STEP", "file" : "${STEP_BIBLE_PATH}/OT_aligned.json", "manuscript" : "TAHOT"},
        ]
    ),
    ("TRANSLATION_LANGUAGES", "", [
                                    "English", 
                                    # "Finnish",
                                    # "Turkish", 
                                    # "Swedish", 
                                    # "Marathi"
                                    ]),
    ("MODELS", "",["CohereForAI/aya-23-8B"]),
    ("OT_DATA", "", "${DATA_DIR}/STEP/OT_aligned.json"),
    ("USE_PRECOMPUTED_EMBEDDINGS", "", False),
)

env = Environment(
    variables=vars,
    # ENV=os.environ,
    tools=[],
    BUILDERS={
        "ConvertFromXML" : Builder(
            action="python scripts/convert_from_xml.py --testament ${TESTAMENT} --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "ConvertFromSTEP" : Builder(
            action="python scripts/convert_from_aligned_step.py --input ${SOURCE} --output ${TARGET} --lang ${LANGUAGE}"
        ),          
        "EmbedDocument" : Builder(
            action="python scripts/embed_document.py --input ${SOURCES[0]} --output ${TARGETS[0]} --batch_size ${BATCH_SIZE}"
        ),
        "MakePrompt": Builder(
            action="python scripts/make_prompt.py --original ${SOURCES[0]} --human_translation ${SOURCES[1]} --src ${SRC_LANG} --tgt ${TGT_LANG} --output ${TARGET}"
        ),
        "TranslateDocument" : Builder(
            action="python scripts/translate_with_sqlite.py --prompt ${PROMPT} --input ${SOURCES[0]} --output ${TARGETS[0]} --model ${MODEL} --batch_size ${BATCH_SIZE}" #${'--vote_threshold ' + str(VOTE_THRESHOLD) if VOTE_THRESHOLD else ''}"
        ),
        "PostProcess": Builder(
            action="python scripts/postprocess.py --inputs ${SOURCES} --output ${TARGET} --src ${SRC_LANG} --tgt ${TGT_LANG}"
        ),
        "ScoreTranslation": Builder(
            action="python scripts/score_translation.py --preds ${SOURCES[0]} --refs ${SOURCES[1]} --sources ${SOURCES[2]} --lang ${TGT_LANG} --output ${TARGET}"
        ),
        "InspectTranslation": Builder(
            action="python scripts/inspect_translations.py --gold ${SOURCES[0]} --embeddings ${SOURCES[1:]} --output ${TARGET}"
        ),
    }
)

# how we decide if a dependency is out of date
env.Decider("timestamp-newer")

## N.B. Not sure if I need the below or if the above takes care of it. 
# this function tells Scons to track the timestamp of the directory rather than content changes (which it can't do for directories), 
# so that a directory can be used as a source or target.
# def dir_timestamp(node, env):
#     return os.path.getmtime(node.abspath)

# # N.B. we pass this either to source_scanner or target_scanner when source or target is a directory.
# scan_timestamp = Scanner(function=dir_timestamp, skeys=['.'])

lang_map = env["LANGUAGE_MAP"]
r_lang_map = {v : k for k, v in lang_map.items()}


        
############### INTT experiments ###############
# keeping track of output files for later analysis
embeddings = {}
human_translations = {}
originals ={}
mt_translations = {}

# Pre-Processing and Embedding HUMAN TRANSLATIONS
HUMAN = []
for testament in ["OT", "NT"]:
    condition_name = "human_translation"
    embeddings[testament] = embeddings.get(testament, {})
    human_translations[testament] = human_translations.get(testament, {})
    
    for language in env["TRANSLATION_LANGUAGES"]:
        manuscript = "JHUBC"
        
        # create the directory structure for the current testament and language
        embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
        embeddings[testament][manuscript][language] = embeddings[testament][manuscript].get(language, {})

        # name the paths for the human translations and embeddings
        human_trans_path = os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, language + ".json.gz")
        emb_path = os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, language + "-embedded.json.gz")
        
        if env["USE_PRECOMPUTED_EMBEDDINGS"]:
            human_trans = env.File(human_trans_path)
            emb = env.File(emb_path)
        else:
            # otherwise, convert the XML files to JSON
            # and embed the human translations
            human_trans = env.ConvertFromXML(
                human_trans_path,
                os.path.join(env["BIBLE_CORPUS"], language + ".xml"),
                TESTAMENT=testament,
            )
            emb = env.EmbedDocument(
                emb_path,
                human_trans,
                BATCH_SIZE=500,                
            )[0]
            HUMAN.append(emb)

        # save the processed translation and embedding
        human_translations[testament][language] = human_trans
        embeddings[testament][manuscript][language][condition_name] = emb

env.Alias(
    "human",
    HUMAN
)

# Pre-Processing and Embedding ORIGINAL DOCUMENTS
ORIGINALS = []
for original in env["ORIGINALS"]:
    condition_name = "original"
    original_language = original["language"]
    original_file = original["file"]
    testament = original["testament"]
    manuscript = "{}-{}".format(original["manuscript"], original_language)
    
    embeddings[testament] = embeddings.get(testament, {})
    embeddings[testament][manuscript] = embeddings[testament].get(manuscript, {})
    embeddings[testament][manuscript][original_language] = embeddings[testament][manuscript].get(original_language, {})

    originals[testament] = originals.get(testament, {})
    originals[testament][manuscript] = originals[testament].get(manuscript, {})
    originals[testament][manuscript][original_language] = embeddings[testament][manuscript].get(original_language, {})

    # name the paths
    orig_path = os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, original_language + ".json.gz")
    orig_emb_path = os.path.join(env["WORK_DIR"], testament, condition_name,  manuscript, original_language + "-embedded.json.gz")
    
    if env["USE_PRECOMPUTED_EMBEDDINGS"]:
        orig = env.File(orig_path)
        orig_emb = env.File(orig_emb_path)
    else:
        orig = env.ConvertFromSTEP(
            orig_path,
            original_file,
            LANGUAGE=original_language
            )

        orig_emb = env.EmbedDocument(
            orig_emb_path,
            orig,
            BATCH_SIZE=500,         
        )[0]

        ORIGINALS.append(orig_emb)
    originals[testament][manuscript][original_language] = orig
    embeddings[testament][manuscript][original_language][condition_name] = orig_emb

    env.Alias(
        "original",
        ORIGINALS
    )
    
    MT = []
    SCORES = []
    # Creating MACHINE TRANSLATIONS from the original documents and embedding them
    for other_language in env["TRANSLATION_LANGUAGES"]:
        condition_name = "machine_translation"
        mt_translations[testament] = mt_translations.get(testament, {})
        mt_translations[testament][manuscript] = mt_translations[testament].get(manuscript, {})
        mt_translations[testament][manuscript][other_language] = mt_translations[testament][manuscript].get(other_language, {})

        embeddings[testament][manuscript][other_language] = embeddings[testament][manuscript].get(other_language, {})        

        src_lang = env["LANGUAGE_MAP"][original_language][3]
        tgt_lang = env["LANGUAGE_MAP"][other_language][3]        

        # grab the appropriate human translation for the current testament and translation language
        human_trans = human_translations[testament][other_language]

        prompt = env.MakePrompt(os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, tgt_lang + ".txt"),
                                [orig, human_trans],
                                SRC_LANG=src_lang, 
                                TGT_LANG=tgt_lang)


        for model in env["MODELS"]:
            model_name = model.replace('/', '_')
            embeddings[testament][manuscript][other_language][model_name] = embeddings[testament][manuscript][other_language].get(model_name, {})
            mt_translations[testament][manuscript][other_language][model_name] = mt_translations[testament][manuscript][other_language].get(model_name, {})


            condition_name = "machine_translation"
            embeddings[testament][manuscript][other_language][condition_name] = embeddings[testament][manuscript][other_language].get(condition_name, {})  

            
            translation_path = os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, model_name, other_language + ".db")
            preds_path = os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, model_name, other_language + ".json")
            emb_path = os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, model_name, other_language + "-embedded.json.gz")
            
            if env["USE_PRECOMPUTED_EMBEDDINGS"]:
                translation = env.File(translation_path)
                preds = env.File(preds_path)
                emb = env.File(emb_path)
            else:
                translation = env.TranslateDocument(
                    translation_path,
                    orig,
                    PROMPT=prompt,
                    BATCH_SIZE=15,
                    MODEL=model,
                )
                # N.B. make sure to create the prompt for the language prior to the translation step.
                env.Depends(translation, prompt)
                env.Depends(translation, orig)

    
                preds = env.PostProcess(
                    preds_path,
                    translation,
                    SRC_LANG=src_lang,
                    TGT_LANG=tgt_lang
                )

                emb = env.EmbedDocument(
                    emb_path,
                    preds,
                    BATCH_SIZE=500,                   
                )[0]
            MT.append(translation)

            # collect the bits you need for translation scoring, comet needs both human refs and sources
            mt_translations[testament][manuscript][other_language][model_name][condition_name] = translation
            human_translation = human_translations[testament][other_language]
            original = originals[testament][manuscript][original_language]

            env.ScoreTranslation(os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, "_".join([other_language, 'score.txt'])),
                                    [preds, human_translation, original],
                                    TGT_LANG=other_language)
            

            embeddings[testament][manuscript][other_language][condition_name]= emb

            ## Now we can get the INTT score  
            intt_score = env.InspectTranslation(
                os.path.join(env["WORK_DIR"], testament, condition_name, manuscript, model_name, other_language + "-score.json"),
                [env["CROSS_REFERENCE_FILE"], [emb]],
            )
            SCORES.append(intt_score)
    # let's make an alias for the machine translations
    env.Alias(
        "mt",
        MT
    )

    env.Alias(
        "score",
        SCORES
    )
                