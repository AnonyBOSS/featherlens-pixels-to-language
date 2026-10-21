from src.data.schemas import SourceSpec


def test_image_ids_include_the_repository_revision_split_and_row():
    train = SourceSpec("Multimodal-Fatima/CUB_train", "train", "a" * 40)
    other_revision = SourceSpec("Multimodal-Fatima/CUB_train", "train", "b" * 40)
    test = SourceSpec("Multimodal-Fatima/CUB_test", "test", "a" * 40)

    assert train.image_id(7) == (
        "hf:Multimodal-Fatima/CUB_train@"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:train:7"
    )
    assert len({train.image_id(7), train.image_id(8), other_revision.image_id(7), test.image_id(7)}) == 4
