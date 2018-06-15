def feature_vector_extraction(candidate):
    """
    :param candidate: a candidate object
    :return: the feature vector
    it consists of three components: img-text, html-text, form-text
    """
    print ("Analyse source and image at:")
    print (candidate.source_html)
    print (candidate.img_path)

    if os.path.exists(candidate.source_html) and os.path.exists(candidate.img_path):
        try:
            img_text = get_img_text_ocr(candidate.img_path)

            #print (img_text)

            if len(img_text) == 0:
                return None

            text_word_str, num_of_forms, attr_word_str = get_structure_html_text(candidate.source_html)
            img_v = text_embedding_into_vector(img_text)
            txt_v = text_embedding_into_vector(text_word_str)
            form_v = text_embedding_into_vector(attr_word_str)
            final_v = img_v + txt_v + form_v + [num_of_forms]
            return final_v

        except:

            print ("error happened! maybe your img/html-source format is not acceptable?")
            return None