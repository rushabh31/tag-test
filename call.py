def __call__(self, text_or_texts: Union[str, List[str], None]) -> Union[List[float], List[List[float]]]:
        """Make the class callable to support direct invocation."""
        # Log the input type to help debug
        logger.info(f"VertexAIEmbeddings called with input type: {type(text_or_texts)}")
        
        # Handle None case
        if text_or_texts is None:
            logger.warning("VertexAIEmbeddings received None input, returning empty embedding")
            return self._get_zero_vector()
            
        # Handle string case
        if isinstance(text_or_texts, str):
            logger.info(f"Processing single string: '{text_or_texts[:30]}...'")
            return self.embed_query(text_or_texts)
            
        # Handle list case
        elif isinstance(text_or_texts, list):
            # Check if list is empty
            if not text_or_texts:
                logger.warning("VertexAIEmbeddings received empty list, returning empty list")
                return []
                
            # Check that all elements are strings
            if all(isinstance(item, str) for item in text_or_texts):
                logger.info(f"Processing list of {len(text_or_texts)} strings")
                return self.embed_documents(text_or_texts)
            else:
                # Log the types of elements
                item_types = [type(item) for item in text_or_texts]
                logger.error(f"List contains non-string elements: {item_types}")
                # Try to convert to strings
                try:
                    str_texts = [str(item) for item in text_or_texts]
                    logger.warning("Converted non-string elements to strings")
                    return self.embed_documents(str_texts)
                except Exception as e:
                    logger.error(f"Failed to convert elements to strings: {e}")
                    raise ValueError(f"Expected list of strings, got list with types: {item_types}")
        else:
            logger.error(f"Unsupported input type: {type(text_or_texts)}")
            raise ValueError(f"Expected string or list of strings, got {type(text_or_texts)}")
