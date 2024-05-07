import json
import numpy as np


def read_document():
    # 读取document数据
    file_in = open(r'./data/multidoc2dial/multidoc2dial_doc.json', 'r', encoding='utf-8')
    # file_in = open(r'/root/code/graph-retrieval-dpr/data/multidoc2dial/multidoc2dial_doc.json', 'r', encoding='utf-8')
    # file_out = open(r'./data/section/candidates.txt', 'w', encoding='utf-8')
    document_whole = json.load(file_in)['doc_data']
    max_node_num = 0
    # 从父亲指向孩子是type 0， 孩子指向父亲是type 1
    # section/title有可能出现重复的情况，在这没有考虑
    graph_content = {}
    domain_to_document_name = {}
    no_title_section_num = 0
    for every_domain in document_whole:
        get_domain_documents = document_whole[every_domain]
        domain_to_document_name[every_domain] = []
        for every_document in get_domain_documents:
            edge_index = []
            edge_type = []
            key_to_id = {}
            id_to_key = {}
            title_to_id = {}
            title_id_to_sections_ids = {}
            section_to_id = {}
            span_id_to_section = {}
            ptr = 0
            document_name = every_document.split('#')[0]
            key_to_id['<1> ' + document_name] = str(ptr)
            id_to_key[str(ptr)] = document_name
            ptr += 1
            all_spans = get_domain_documents[every_document]['spans']
            for every_span_index in all_spans:
                get_parent = all_spans[every_span_index]['title'].replace('\n', '').replace('\t', '').strip(
                    ' ')
                get_data = all_spans[every_span_index]['text_sp'].replace('\n', '').replace('\t', '').strip(
                    ' ')
                if len(get_parent) == 0:
                    no_title_section_num += 1
                if ('<1> ' + document_name + ' <2> ' + get_parent not in key_to_id) or (get_data == get_parent):
                    # title
                    get_data = '<1> ' + document_name + ' <2> ' + get_data
                    get_parent = '<1> ' + document_name + ' <2> ' + get_parent
                    # title
                    title_to_id[get_data] = ptr
                    key_to_id[get_data] = ptr
                    id_to_key[str(ptr)] = get_data
                    edge_index.append([0, ptr])
                    edge_type.append(0)
                    edge_index.append([ptr, 0])
                    edge_type.append(1)
                    span_id_to_section[all_spans[every_span_index]['id_sp']] = get_data
                    ptr += 1
                else:
                    get_section = all_spans[every_span_index]['text_sec'].replace('\n', '').replace('\t', '').strip(
                        ' ')
                    get_section = '<1> ' + document_name + ' <2> ' + get_parent + ' <3> ' + get_section
                    get_parent = '<1> ' + document_name + ' <2> ' + get_parent
                    # section
                    if get_section not in key_to_id:
                        # file_out.write(get_section + '\n')
                        key_to_id[get_section] = ptr
                        id_to_key[str(ptr)] = get_section
                        section_to_id[get_section] = ptr
                        now_pid = title_to_id[get_parent]
                        edge_index.append([now_pid, ptr])
                        edge_type.append(0)
                        edge_index.append([ptr, now_pid])
                        edge_type.append(1)
                        if get_parent not in title_id_to_sections_ids:
                            title_id_to_sections_ids[get_parent] = [ptr]
                        else:
                            title_id_to_sections_ids[get_parent].append(ptr)
                        ptr += 1
                    span_id_to_section[all_spans[every_span_index]['id_sp']] = get_section
            node_num = len(key_to_id)
            if node_num > max_node_num:
                max_node_num = node_num
            graph_content[every_document] = {
                'edge_index': np.array([np.array(edge_index)[:, 0], np.array(edge_index)[:, 1]]),
                'edge_type': edge_type, 'content': key_to_id, 'span_id_to_section': span_id_to_section,
                'title_to_id': title_to_id, 'title_id_to_sections_ids': title_id_to_sections_ids,
                'id_to_content': id_to_key, 'domain': every_domain
            }
            domain_to_document_name[every_domain].append(every_document)
    print(no_title_section_num)
    return graph_content, domain_to_document_name

read_document()
# graph_content, max_node_num = read_document()
