from qdrant_client import QdrantClient
from io import BytesIO
import streamlit as st
import base64

# 1. Define the Qdrant collection name that we used to
# store all of our metadata and vectors in
collection_name = "Egypt-Dakahlia-Mansoura"

# 2. Set up a state variable that we'll re-use throughout
# the rest of the app
if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None


def set_selected_record(new_record):
    # 3. Create a function that allows us to easily set the
    # "selected_record" value. We'll use this later.
    st.session_state.selected_record = new_record


@st.cache_resource
def get_client():
    # 4. Create the Qdrant client. These must be set up in
    # the .streamlit/secrets.toml file.
    return QdrantClient(
        url="",
        api_key="")


def get_initial_records():
    # 5. When the app first starts, let's show
    # a small sample of images to the user
    # the first 12.
    client = get_client()

    records, _ = client.scroll(
        collection_name=collection_name,
        with_vectors=False,
        limit=120
    )

    return records


def get_similar_records():
    # 6. If the user has selected a record that they want
    # to see similar items to, then this is the function
    # that will be used for that
    client = get_client()

    if st.session_state.selected_record is not None:
        return client.recommend(
            collection_name=collection_name,
            positive=[st.session_state.selected_record.id],
            limit=6
        )
    return records


def get_bytes_from_base64(base64_string):
    # 7. Define a Convenience function to convert base64 back
    # into bytes that can be used by Streamlit to render Images
    return BytesIO(base64.b64decode(base64_string))


# 8. Get the records. This function will be re-called multiple
# times throughout the lifecycle of our app. We fetch the original
# records if there is nothing selected otherwise we'll fetch the
# recommendations.
records = get_similar_records(
) if st.session_state.selected_record is not None else get_initial_records()

# 9. If we have a selected record, then show that image
# at the top of the screen
if st.session_state.selected_record:
    image_bytes = get_bytes_from_base64(
        st.session_state.selected_record.payload["base64"])
    st.header("Images similar to:")
    st.image(
        image=image_bytes
    )
    st.divider()

# 10. Set up the grid that we'll use to render our images into
column = st.columns(3)

# 11. Iterate over all of the fetched records from the DB,
# and render a preview of each image using the base64 string
# in the document's payload.
for idx, record in enumerate(records):
    col_idx = idx % 3
    image_bytes = get_bytes_from_base64(record.payload["base64"])
    with column[col_idx]:
        st.image(
            image=image_bytes
        )
        st.button(
            label="Find similar image",
            key=record.id,
            on_click=set_selected_record,
            args=[record]
        )
